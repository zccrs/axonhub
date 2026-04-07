package claw

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/looplj/axonhub/axon/agent"
	"github.com/looplj/axonhub/axon/bus"
	"github.com/looplj/axonhub/axon/subagent"
	"github.com/looplj/axonhub/axon/tools"
)

const (
	SummarizerAgentName = "axonclaw_summarizer"

	compactInstruction = `

CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
Tool calls will be REJECTED and will waste your only turn — you will fail the task.

---

## Context Compaction Task

The conversation history above has grown too large and needs to be compacted.

Your task: Create a concise summary that preserves critical information for continuing the work.

### Analysis Phase
First, analyze the conversation and identify:
- Key decisions made and their rationale
- File changes (created, modified, deleted) with brief descriptions
- User preferences and coding style discovered
- Important entities (people, projects, concepts, tools)
- Task progress and current status
- Errors encountered and how they were resolved

### Summary Output
Return a concise plain-text summary (NOT JSON) that covers:

1. **Primary Request**: What the user originally asked for
2. **Key Decisions**: Important choices made and why
3. **Files Changed**: List of files created/modified/deleted with brief descriptions
4. **Current State**: What is actively being worked on right now
5. **Pending Items**: Any unfinished tasks or next steps
6. **Important Context**: User preferences, constraints, or other critical info

The summary will be injected into the conversation context to help continue the work.
Keep it focused and concise — omit routine tool interactions and focus on what matters for continuing the work.

Do NOT call any tools. Respond with TEXT ONLY.`
)

func SummarizerDefinition() *subagent.Definition {
	return &subagent.Definition{
		Name:        SummarizerAgentName,
		Hidden:      true,
		Description: "Context compaction summarizer",
		Tools:       map[string]bool{},
	}
}

type Summarizer interface {
	Summarize(ctx context.Context, messages []agent.Message) (string, error)
}

type ForkedCompactSummarizer struct {
	agent       *agent.Agent
	provider    agent.Provider
	model       string
	logger      *slog.Logger
	bus         bus.EventBus
	middlewares []agent.Middleware
}

type ForkedCompactSummarizerOptions struct {
	Agent       *agent.Agent
	Provider    agent.Provider
	Model       string
	Logger      *slog.Logger
	Bus         bus.EventBus
	Middlewares []agent.Middleware
}

func NewForkedCompactSummarizer(opts ForkedCompactSummarizerOptions) *ForkedCompactSummarizer {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &ForkedCompactSummarizer{
		agent:       opts.Agent,
		provider:    opts.Provider,
		model:       opts.Model,
		logger:      logger,
		bus:         opts.Bus,
		middlewares: opts.Middlewares,
	}
}

func (s *ForkedCompactSummarizer) Summarize(ctx context.Context, messages []agent.Message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	cfg := s.agent.Config()

	model := cfg.Model
	if s.model != "" {
		model = s.model
	}

	toolDefs := s.agent.RegisteredTools()

	toolDefinitions := make([]agent.ToolDefinition, len(toolDefs))
	for i, t := range toolDefs {
		toolDefinitions[i] = t.Definition()
	}

	forkCfg := agent.Config{
		Model:         model,
		MaxIterations: 1,
		SystemPrompts: cfg.SystemPrompts,
	}

	forkOpts := []agent.Option{
		agent.WithLogger(s.logger.With("component", "compact_fork")),
		agent.WithMessages(messages),
	}

	if s.bus != nil {
		forkOpts = append(forkOpts, agent.WithBus(s.bus))
	}

	if len(s.middlewares) > 0 {
		forkOpts = append(forkOpts, agent.WithMiddlewares(s.middlewares...))
	}

	forkAgent := agent.New(forkCfg, s.provider, forkOpts...)

	for _, t := range toolDefs {
		forkAgent.RegisterTool(t)
	}

	compactPrompt := compactInstruction

	result, err := forkAgent.Process(ctx, agent.Content{Text: &compactPrompt})
	if err != nil {
		return "", fmt.Errorf("compact fork failed: %w", err)
	}

	if result.Output == "" {
		return "", fmt.Errorf("compact fork returned empty output")
	}

	return strings.TrimSpace(result.Output), nil
}

type SmartSummarizer struct {
	manager     *subagent.Manager
	provider    agent.Provider
	model       string
	skillMgr    *tools.SkillManager
	workspace   string
	bus         bus.EventBus
	middlewares []agent.Middleware
	logger      *slog.Logger
}

type SmartSummarizerOptions struct {
	Manager     *subagent.Manager
	Provider    agent.Provider
	Model       string
	SkillMgr    *tools.SkillManager
	Workspace   string
	Bus         bus.EventBus
	Middlewares []agent.Middleware
	Logger      *slog.Logger
}

func NewSmartSummarizer(opts SmartSummarizerOptions) *SmartSummarizer {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &SmartSummarizer{
		manager:     opts.Manager,
		provider:    opts.Provider,
		model:       opts.Model,
		skillMgr:    opts.SkillMgr,
		workspace:   opts.Workspace,
		bus:         opts.Bus,
		middlewares: opts.Middlewares,
		logger:      logger,
	}
}

func (s *SmartSummarizer) Summarize(ctx context.Context, messages []agent.Message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	conversationText := RenderMessages(messages, MessageRenderOptions{Separator: "\n\n"})

	def, ok := s.manager.Get(SummarizerAgentName)
	if !ok {
		return "", fmt.Errorf("summarizer subagent definition %q not found", SummarizerAgentName)
	}

	model := def.Model
	if model == "" {
		model = s.model
	}

	allowedTools, deniedTools := subagent.BuildToolFiltersFromDefinition(def.Tools)

	task := fmt.Sprintf("Please summarize the following conversation:\n\n%s", conversationText)

	toolSource := s.buildToolSource()

	result, err := subagent.Run(ctx, subagent.Config{
		Model:         model,
		SystemPrompts: []string{def.Description},
		AllowedTools:  allowedTools,
		DeniedTools:   deniedTools,
		Provider:      s.provider,
		Bus:           s.bus,
		Middlewares:   s.middlewares,
		Logger:        s.logger.With("component", "summarizer_subagent"),
	}, task, toolSource)
	if err != nil {
		return "", fmt.Errorf("summarizer subagent failed: %w", err)
	}

	if result.Output == "" {
		return "", fmt.Errorf("summarizer subagent returned empty output")
	}

	return result.Output, nil
}

func (s *SmartSummarizer) buildToolSource() subagent.ToolSource {
	agentTools := []agent.Tool{}

	agentTools = append(agentTools, tools.NewAgentTool(tools.NewReadTool(s.workspace, true)))
	agentTools = append(agentTools, tools.NewAgentTool(tools.NewWriteTool(s.workspace, true)))
	agentTools = append(agentTools, tools.NewAgentTool(tools.NewBashTool(s.workspace, true, true)))
	agentTools = append(agentTools, tools.NewAgentTool(tools.NewGrepTool(s.workspace, true)))
	agentTools = append(agentTools, tools.NewAgentTool(tools.NewGlobTool(s.workspace, true)))

	if s.skillMgr != nil {
		agentTools = append(agentTools, tools.NewAgentTool(tools.NewSkillTool(s.skillMgr)))
	}

	return &staticToolSource{tools: agentTools}
}

type staticToolSource struct {
	tools []agent.Tool
}

func (s *staticToolSource) AvailableTools() []agent.Tool    { return s.tools }
func (s *staticToolSource) Middlewares() []agent.Middleware { return nil }
