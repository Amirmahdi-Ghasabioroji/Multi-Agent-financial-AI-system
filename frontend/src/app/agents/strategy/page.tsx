import { AgentWorkspace } from "@/components/agent-workspace";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "Strategy Workbench" };

export default function StrategyPage() {
  return <AgentWorkspace kind="strategy" />;
}
