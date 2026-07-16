import { AgentWorkspace } from "@/components/agent-workspace";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "Execution Workbench" };

export default function ExecutionPage() {
  return <AgentWorkspace kind="execution" />;
}
