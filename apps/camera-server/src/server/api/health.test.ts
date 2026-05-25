import { describe, expect, test } from "bun:test";
import type { Request, Response } from "express";
import { getHealth } from "./health";

describe("getHealth", () => {
  test("returns ok status with ISO timestamp", () => {
    const json = { status: "", timestamp: "" };
    const res = {
      json(payload: { status: string; timestamp: string }) {
        json.status = payload.status;
        json.timestamp = payload.timestamp;
      },
    } as Response;

    getHealth({} as Request, res);

    expect(json.status).toBe("ok");
    expect(() => new Date(json.timestamp)).not.toThrow();
    expect(Number.isNaN(new Date(json.timestamp).getTime())).toBe(false);
  });
});
