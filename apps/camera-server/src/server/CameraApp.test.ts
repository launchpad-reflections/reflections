import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import express, { type Express } from "express";
import type { AddressInfo } from "node:net";
import type { Server } from "node:http";

type MockSession = {
  audio: { speak: ReturnType<typeof mock> };
};

let expressApp: Express;

mock.module("@mentra/sdk", () => ({
  AppServer: class {
    protected app = express();

    constructor(_opts: unknown) {
      this.app.use(express.json());
      expressApp = this.app;
    }

    getExpressApp() {
      return this.app;
    }

    addCleanupHandler(_handler: () => void) {}
  },
  AppSession: class {},
}));

const { CameraApp } = await import("./CameraApp");

function getActiveMap(app: InstanceType<typeof CameraApp>): Map<string, MockSession> {
  return (app as unknown as { active: Map<string, MockSession> }).active;
}

describe("CameraApp POST /speak", () => {
  let app: InstanceType<typeof CameraApp>;
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    app = new CameraApp({ packageName: "test.pkg", apiKey: "test-key", port: 0 });
    await new Promise<void>((resolve) => {
      server = expressApp.listen(0, "127.0.0.1", () => resolve());
    });
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;
  });

  afterEach(async () => {
    getActiveMap(app).clear();
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  });

  test("400 when text is missing", async () => {
    const res = await fetch(`${baseUrl}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(400);
    expect(await res.json()).toEqual({ error: "text (non-empty string) is required" });
  });

  test("400 when text is whitespace only", async () => {
    const res = await fetch(`${baseUrl}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "   " }),
    });
    expect(res.status).toBe(400);
  });

  test("404 when no active glasses session", async () => {
    const res = await fetch(`${baseUrl}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "hello" }),
    });
    expect(res.status).toBe(404);
    expect(await res.json()).toEqual({ error: "no active glasses session" });
  });

  test("200 and speaks through every active session", async () => {
    const speakA = mock(async (_text: string) => {});
    const speakB = mock(async (_text: string) => {});
    getActiveMap(app).set("user-a", { audio: { speak: speakA } });
    getActiveMap(app).set("user-b", { audio: { speak: speakB } });

    const res = await fetch(`${baseUrl}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "  hello there  " }),
    });

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ success: true, sessions: 2 });
    expect(speakA).toHaveBeenCalledTimes(1);
    expect(speakB).toHaveBeenCalledTimes(1);
    expect(speakA.mock.calls[0]?.[0]).toBe("  hello there  ");
  });

  test("500 when speak throws", async () => {
    getActiveMap(app).set("user-a", {
      audio: {
        speak: mock(async () => {
          throw new Error("tts failed");
        }),
      },
    });

    const res = await fetch(`${baseUrl}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "boom" }),
    });

    expect(res.status).toBe(500);
    expect(await res.json()).toEqual({ error: "tts failed" });
  });

  test("400 when text exceeds max length", async () => {
    getActiveMap(app).set("user-a", { audio: { speak: mock(async () => {}) } });

    const res = await fetch(`${baseUrl}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "x".repeat(501) }),
    });

    expect(res.status).toBe(400);
    const body = (await res.json()) as { error: string };
    expect(body.error).toMatch(/at most 500 characters/);
  });

  test("400 when JSON body is malformed", async () => {
    const res = await fetch(`${baseUrl}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{not-json",
    });

    expect([400]).toContain(res.status);
  });
});

describe("CameraApp GET /health", () => {
  let app: InstanceType<typeof CameraApp>;
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    app = new CameraApp({ packageName: "test.pkg", apiKey: "test-key", port: 0 });
    await new Promise<void>((resolve) => {
      server = expressApp.listen(0, "127.0.0.1", () => resolve());
    });
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;
  });

  afterEach(async () => {
    getActiveMap(app).clear();
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  });

  test("returns 200 with status: ok", async () => {
    const res = await fetch(`${baseUrl}/health`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { status: string; timestamp: string };
    expect(body.status).toBe("ok");
    expect(typeof body.timestamp).toBe("string");
  });
});
