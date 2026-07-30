import { AgentWorkspace } from "@/components/agent-workspace";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "Analyst Workbench" };

export default function AnalystPage() {
  return <AgentWorkspace kind="analyst" />;
}
