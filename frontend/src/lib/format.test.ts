import {
  jobLabel,
  jobQuery,
  jobTickers,
  parseJsonObject,
  percent,
  titleCase,
} from "@/lib/format";
import { describe, expect, it } from "vitest";

describe("format helpers", () => {
  it("validates advanced JSON as an object", () => {
    expect(parseJsonObject('{"ticker":"AAPL"}')).toEqual({
      value: { ticker: "AAPL" },
    });
    expect(parseJsonObject("[1,2]")).toEqual({
      error: 'Enter a JSON object, for example { "ticker": "AAPL" }.',
    });
    expect(parseJsonObject("{")).toHaveProperty("error");
  });

  it("formats labels and normalized percentages", () => {
    expect(titleCase("risk_on")).toBe("Risk On");
    expect(percent(0.725)).toBe("73%");
    expect(percent(8.2)).toBe("8.2%");
  });

  it("chooses the most useful job label", () => {
    expect(jobLabel({ query: "Assess margins", ticker: "AAPL" })).toBe(
      "Assess margins",
    );
    expect(jobLabel({ ticker: "MSFT" })).toBe("MSFT");
    expect(jobLabel({ kind: "pipeline" })).toBe("Pipeline");
  });

  it("reads stored query and tickers from job payload/result", () => {
    const job = {
      payload: {
        query: "How is the rate outlook?",
        tickers: ["aapl", "MSFT"],
      },
      result: {
        query: "Rewritten query",
        risk: { universe: ["NVDA"] },
      },
    };
    expect(jobQuery(job)).toBe("How is the rate outlook?");
    expect(jobTickers(job)).toEqual(["AAPL", "MSFT"]);
    expect(jobLabel(job)).toBe("How is the rate outlook?");
  });
});
