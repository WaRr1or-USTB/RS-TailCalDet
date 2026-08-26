import {describe, expect, it} from "vitest";

import {broadClassFor, filterDetections} from "./model";
import type {ViewerDetection} from "./types";

const detection = (id: string, classId: number, confidence: number | null): ViewerDetection => ({
  id,
  classId,
  className: id,
  broadClass: broadClassFor(classId),
  confidence,
  bbox: {x1: 0, y1: 0, x2: 10, y2: 10, width: 10, height: 10},
  sourceTileIds: [],
  sourceDetectionIds: [],
  stage: confidence === null ? "ground_truth" : "final",
});

describe("workbench detection model", () => {
  it("maps the audited fine classes into broad categories", () => {
    expect(broadClassFor(0)).toBe("ship");
    expect(broadClassFor(23)).toBe("aircraft");
    expect(broadClassFor(24)).toBe("vehicle");
  });

  it("applies confidence, broad category and fine-class filters together", () => {
    const rows = [detection("ship", 3, 0.91), detection("aircraft", 4, 0.72), detection("vehicle", 24, 0.95)];
    expect(filterDetections(rows, 0.8, "all", new Set())).toHaveLength(2);
    expect(filterDetections(rows, 0, "aircraft", new Set()).map((item) => item.id)).toEqual(["aircraft"]);
    expect(filterDetections(rows, 0.8, "all", new Set([24])).map((item) => item.id)).toEqual(["vehicle"]);
  });

  it("does not hide ground truth when a confidence filter is active", () => {
    expect(filterDetections([detection("gt", 0, null)], 0.99, "all", new Set())).toHaveLength(1);
  });
});
