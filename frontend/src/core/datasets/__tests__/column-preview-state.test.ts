/* Copyright 2026 Marimo. All rights reserved. */
import { describe, expect, it } from "vitest";
import type { DataColumnPreview } from "@/core/kernel/messages";
import { exportedForTesting } from "../state";

const { reducer, initialState } = exportedForTesting;

function preview(
  requestId: string | null,
  table = "db.schema.users",
  column = "email",
): DataColumnPreview {
  return {
    request_id: requestId,
    table_name: table,
    column_name: column,
    chart_spec: null,
    chart_code: null,
    error: null,
    missing_packages: null,
    stats: null,
  };
}

describe("column preview staleness", () => {
  it("applies a preview that matches the latest tracked request", () => {
    const tracked = reducer(initialState(), {
      type: "trackColumnPreviewRequest",
      payload: { tableColumn: "db.schema.users:email", requestId: "r1" },
    });

    const next = reducer(tracked, {
      type: "addColumnPreview",
      payload: preview("r1"),
    });

    expect(next.columnsPreviews.get("db.schema.users:email")).toEqual(
      preview("r1"),
    );
  });

  it("drops a stale preview after a newer request was made", () => {
    const tracked = reducer(initialState(), {
      type: "trackColumnPreviewRequest",
      payload: { tableColumn: "db.schema.users:email", requestId: "r2" },
    });

    // Response to the previous request arrives late
    const next = reducer(tracked, {
      type: "addColumnPreview",
      payload: preview("r1"),
    });

    expect(next.columnsPreviews.get("db.schema.users:email")).toBeUndefined();
  });

  it("still applies untracked previews for backward compatibility", () => {
    const next = reducer(initialState(), {
      type: "addColumnPreview",
      payload: preview(null, "tbl", "id"),
    });

    expect(next.columnsPreviews.get("tbl:id")).toEqual(
      preview(null, "tbl", "id"),
    );
  });

  it("keeps the newest response after retries", () => {
    let state = reducer(initialState(), {
      type: "trackColumnPreviewRequest",
      payload: { tableColumn: "db.schema.users:email", requestId: "r1" },
    });
    state = reducer(state, {
      type: "trackColumnPreviewRequest",
      payload: { tableColumn: "db.schema.users:email", requestId: "r2" },
    });

    state = reducer(state, {
      type: "addColumnPreview",
      payload: { ...preview("r2"), error: "ok-after-retry" },
    });

    expect(state.columnsPreviews.get("db.schema.users:email")?.error).toBe(
      "ok-after-retry",
    );
  });
});
