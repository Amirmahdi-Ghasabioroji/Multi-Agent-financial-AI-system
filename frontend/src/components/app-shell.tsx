"use client";

import { api } from "@/lib/api";
import type { AppConfig, Health } from "@/lib/types";
import clsx from "clsx";
import {
  Activity,
  Bot,
  BrainCircuit,
  ChartNoAxesCombined,
  ChevronLeft,
  CircleGauge,
  ClipboardCheck,
  Database,
  FileClock,
  Menu,
  Network,
  PanelLeft,
  Scale,
  ShieldCheck,
  Sparkles,
  Target,
  X,
  Zap,
  type LucideIcon,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";
import { ModeBadge, StatusBadge } from "@/components/ui";

const primaryNav: Array<{ href: string; label: string; icon: LucideIcon }> = [
  { href: "/", label: "Overview", icon: CircleGauge },
  { href: "/workspace", label: "Pipeline workspace", icon: Sparkles },
];

const agentNav: Array<{ href: string; label: string; icon: LucideIcon }> = [
  { href: "/agents/analyst", label: "Analyst", icon: BrainCircuit },
  { href: "/agents/risk", label: "Risk", icon: ShieldCheck },
  { href: "/agents/strategy", label: "Strategy", icon: Target },
  { href: "/agents/execution", label: "Execution", icon: Zap },
];

const utilityNav: Array<{ href: string; label: string; icon: LucideIcon }> = [
  { href: "/history", label: "Run history", icon: FileClock },
  { href: "/evaluation", label: "Evaluation reports", icon: ClipboardCheck },
  { href: "/data", label: "Data & corpus", icon: Database },
];

function NavItem({
  href,
  label,
  icon: Icon,
  collapsed,
  onClick,
}: {
  href: string;
  label: string;
  icon: LucideIcon;
  collapsed: boolean;
  onClick: () => void;
}) {
  const pathname = usePathname();
  const active =
    href === "/" ? pathname === href : pathname === href || pathname.startsWith(`${href}/`);
  return (
    <Link
      href={href}
      onClick={onClick}
      className={clsx("nav-item", active && "nav-item-active")}
      aria-current={active ? "page" : undefined}
      title={collapsed ? label : undefined}
    >
      <Icon size={18} strokeWidth={1.8} aria-hidden />
      {!collapsed && <span>{label}</span>}
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [health, setHealth] = useState<Health>();
  const [config, setConfig] = useState<AppConfig>();
  const [reachable, setReachable] = useState<boolean>();

  useEffect(() => {
    let active = true;
    const load = async () => {
      const [healthResult, configResult] = await Promise.allSettled([
        api.health(),
        api.config(),
      ]);
      if (!active) return;
      if (healthResult.status === "fulfilled") {
        setHealth(healthResult.value);
        setReachable(true);
      } else {
        setReachable(false);
      }
      if (configResult.status === "fulfilled") setConfig(configResult.value);
    };
    void load();
    const interval = window.setInterval(load, 30_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const isDemo =
    config?.demo_mode ?? health?.mode?.toLowerCase() !== "live";

  return (
    <div className="app-shell">
      {mobileOpen && (
        <button
          className="sidebar-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside
        className={clsx(
          "sidebar",
          collapsed && "sidebar-collapsed",
          mobileOpen && "sidebar-mobile-open",
        )}
        aria-label="Primary navigation"
      >
        <div className="brand">
          <div className="brand-mark" aria-hidden>
            <ChartNoAxesCombined size={22} />
          </div>
          {!collapsed && (
            <div>
              <strong>MAFAS</strong>
              <span>Research workstation</span>
            </div>
          )}
          <button
            className="icon-button mobile-close"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        <nav className="nav-groups">
          <div className="nav-group">
            {!collapsed && <p>Command centre</p>}
            {primaryNav.map((item) => (
              <NavItem
                {...item}
                collapsed={collapsed}
                key={item.href}
                onClick={() => setMobileOpen(false)}
              />
            ))}
          </div>
          <div className="nav-group">
            {!collapsed && <p>Specialists</p>}
            {agentNav.map((item) => (
              <NavItem
                {...item}
                collapsed={collapsed}
                key={item.href}
                onClick={() => setMobileOpen(false)}
              />
            ))}
          </div>
          <div className="nav-group">
            {!collapsed && <p>Operations</p>}
            {utilityNav.map((item) => (
              <NavItem
                {...item}
                collapsed={collapsed}
                key={item.href}
                onClick={() => setMobileOpen(false)}
              />
            ))}
          </div>
        </nav>

        <div className="sidebar-footer">
          {!collapsed && (
            <div className="sidebar-system">
              <Network size={16} aria-hidden />
              <div>
                <span>Pipeline topology</span>
                <small>4-agent sequential route</small>
              </div>
            </div>
          )}
          <button
            className="collapse-button"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <PanelLeft size={18} /> : <ChevronLeft size={18} />}
            {!collapsed && <span>Collapse</span>}
          </button>
        </div>
      </aside>

      <div className="app-column">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={() => setMobileOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={20} />
          </button>
          <div className="topbar-title">
            <Activity size={16} aria-hidden />
            <span>Multi-Agent Financial Analysis System</span>
          </div>
          <div className="topbar-status" aria-label="System status">
            <ModeBadge demo={isDemo} />
            <StatusBadge
              status={
                reachable === undefined
                  ? "Checking"
                  : reachable
                    ? health?.status ?? "Online"
                    : "Offline"
              }
            />
          </div>
        </header>

        <main className="main-content">{children}</main>
        <footer className="disclaimer">
          <Scale size={15} aria-hidden />
          <span>
            Research and simulation only. Outputs are probabilistic, may be
            incomplete, and are not financial advice or a trading instruction.
          </span>
          <Bot size={15} aria-hidden />
        </footer>
      </div>
    </div>
  );
}
