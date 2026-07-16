import { AgentWorkspace } from "@/components/agent-workspace";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "Risk Workbench" };

export default function RiskPage() {
  return <AgentWorkspace kind="risk" />;
}
