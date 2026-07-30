import { render, screen } from "@testing-library/react";
import { EvaluationReport } from "@/components/evaluation-report";

describe("EvaluationReport", () => {
  it("renders suite metrics and methodology notes", () => {
    render(
      <EvaluationReport
        report={{
          generated_at: "2026-07-30T12:00:00+00:00",
          summary_metrics: [
            {
              name: "simulation_mean_calibration_error",
              label: "Monte Carlo: Mean calibration error",
              value: 0.04,
              unit: "ratio",
            },
          ],
          suites: [
            {
              suite: "simulation",
              label: "Monte Carlo calibration",
              status: "completed",
              duration_ms: 42,
              metrics: [
                {
                  name: "mean_calibration_error",
                  label: "Mean calibration error",
                  value: 0.04,
                  unit: "ratio",
                },
              ],
              cases: [
                {
                  id: "trend_long",
                  label: "Uptrend long setup",
                  metrics: [
                    {
                      name: "calibration_error",
                      label: "Calibration error",
                      value: 0.03,
                      unit: "ratio",
                    },
                  ],
                  notes: "120 empirical paths",
                },
              ],
            },
          ],
          notes: ["Monte Carlo calibration compares bootstrap probabilities to walk-forward outcomes."],
        }}
      />,
    );

    expect(screen.getByText("System quality metrics")).toBeInTheDocument();
    expect(screen.getByText("Monte Carlo calibration")).toBeInTheDocument();
    expect(screen.getByText("Uptrend long setup")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Monte Carlo calibration compares bootstrap probabilities to walk-forward outcomes.",
      ),
    ).toBeInTheDocument();
  });
});
