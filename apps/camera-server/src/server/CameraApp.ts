import { AppServer, AppSession } from "@mentra/sdk";
import type { Request, Response } from "express";
import { getHealth } from "./api/health";
import { sessions } from "./manager/SessionManager";

const WHIP_URL = process.env.WHIP_URL;

// Hard ceiling on /speak payload size. The SDK forwards `text` to the
// glasses TTS engine; values much longer than a sentence or two never
// make sense and can DOS the device if a misbehaving client floods us.
const MAX_SPEAK_CHARS = 500;
// Periodic SSE heartbeat to keep proxies and the Python viewer's keep-alive
// timers happy even when no transcription events are flowing.
const SSE_HEARTBEAT_MS = 25_000;

export interface CameraAppConfig {
  packageName: string;
  apiKey: string;
  port: number;
}

export class CameraApp extends AppServer {
  private active = new Map<string, AppSession>();
  private stopping = new Set<string>();
  private transcriptClients = new Set<Response>();

  constructor(config: CameraAppConfig) {
    super({
      packageName: config.packageName,
      apiKey: config.apiKey,
      port: config.port,
    });

    const expressApp = this.getExpressApp();

    expressApp.get("/health", getHealth);

    expressApp.get("/transcripts", (req: Request, res: Response) => {
      res.set({
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      });
      res.flushHeaders();
      res.write(": connected\n\n");
      this.transcriptClients.add(res);

      const heartbeat = setInterval(() => {
        try {
          res.write(": ping\n\n");
        } catch {
          // Write failure → the cleanup below will drop the client.
        }
      }, SSE_HEARTBEAT_MS);
      // Don't keep the event loop alive for a client that has gone away.
      heartbeat.unref?.();

      const cleanup = () => {
        clearInterval(heartbeat);
        this.transcriptClients.delete(res);
      };
      req.on("close", cleanup);
      req.on("error", cleanup);
      res.on("close", cleanup);
      res.on("error", cleanup);
    });

    // POST /speak — body: {"text": "..."}. Plays TTS through the glasses
    // speaker of every active session. Used by the proactivity pipeline
    // and the Python viewer (`apps/viewer/cli.py`). The MentraOS SDK
    // registers express.json() upstream so req.body is already parsed —
    // we also fall back to reading the raw stream in case the middleware
    // is ever changed.
    expressApp.post("/speak", async (req: Request, res: Response) => {
      try {
        let text: unknown;
        if (req.body && typeof req.body === "object") {
          text = req.body.text;
        } else if (typeof req.body === "string") {
          try {
            text = JSON.parse(req.body).text;
          } catch {
            return res.status(400).json({ error: "invalid JSON body" });
          }
        } else {
          // Last-resort: drain the raw stream (only happens if no json
          // body-parser middleware is active).
          const raw = await new Promise<string>((resolve, reject) => {
            let buf = "";
            req.setEncoding?.("utf8");
            req.on("data", (chunk: string) => {
              buf += chunk;
            });
            req.on("end", () => resolve(buf));
            req.on("error", reject);
          });
          try {
            text = JSON.parse(raw || "{}").text;
          } catch {
            return res.status(400).json({ error: "invalid JSON body" });
          }
        }

        if (typeof text !== "string" || !text.trim()) {
          return res.status(400).json({ error: "text (non-empty string) is required" });
        }
        if (text.length > MAX_SPEAK_CHARS) {
          return res.status(400).json({
            error: `text must be at most ${MAX_SPEAK_CHARS} characters`,
          });
        }
        if (this.active.size === 0) {
          return res.status(404).json({ error: "no active glasses session" });
        }

        await Promise.all(
          Array.from(this.active.values()).map((s) => s.audio.speak(text as string)),
        );
        console.log(`🔊 speak: ${text}`);
        res.json({ success: true, sessions: this.active.size });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        console.error("speak error:", err);
        res.status(500).json({ error: message });
      }
    });
  }

  private broadcastTranscript(payload: object): void {
    const line = `data: ${JSON.stringify(payload)}\n\n`;
    for (const res of this.transcriptClients) {
      let ok = false;
      try {
        ok = res.write(line);
      } catch {
        this.transcriptClients.delete(res);
        continue;
      }
      if (!ok) {
        // Slow client: drop it rather than buffer transcripts unboundedly.
        try {
          res.end();
        } catch {
          // Best-effort close; ignore secondary errors.
        }
        this.transcriptClients.delete(res);
      }
    }
  }

  protected async onSession(
    session: AppSession,
    sessionId: string,
    userId: string,
  ): Promise<void> {
    console.log(`📸 Camera session started for ${userId}`);
    this.active.set(userId, session);

    const user = sessions.getOrCreate(userId);
    user.setAppSession(session);

    if (!WHIP_URL) {
      console.warn("⚠ WHIP_URL is not set; not starting stream.");
      return;
    }

    const startStream = async () => {
      try {
        console.log(`🎥 Starting WHIP stream to ${WHIP_URL}`);
        await session.camera.startStream({
          rtmpUrl: WHIP_URL,
          video: {
            width: 960,
            height: 540,
            frameRate: 15,
            bitrate: 2_000_000,
          },
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
          },
          stream: { durationLimit: 7200 },
        });
        console.log(`✅ Stream request sent`);
      } catch (err) {
        console.error(`❌ Failed to start stream:`, err);
      }
    };

    const transcriptionCleanup = session.events.onTranscription((data) => {
      const text = (data.text ?? "").trim();
      if (!text) return;
      this.broadcastTranscript({
        userId,
        text,
        isFinal: data.isFinal,
        startTime: data.startTime,
        endTime: data.endTime,
        speakerId: data.speakerId,
        language: data.detectedLanguage ?? data.transcribeLanguage,
      });
    });
    this.addCleanupHandler(transcriptionCleanup);

    session.camera.onStreamStatus((s) => {
      console.log(`📹 stream ${s.status}`, s.errorDetails ?? "");
      if (s.status === "timeout" || s.status === "disconnected" || s.status === "error") {
        if (this.stopping.has(userId)) return;
        console.log(`🔄 Stream ended with ${s.status}, restarting in 3s...`);
        setTimeout(() => {
          if (!this.stopping.has(userId)) startStream();
        }, 3000);
      }
    });

    await startStream();
  }

  protected async onStop(
    sessionId: string,
    userId: string,
    reason: string,
  ): Promise<void> {
    console.log(`👋 Session ending for ${userId} (reason: ${reason})`);
    this.stopping.add(userId);

    const session = this.active.get(userId);
    if (session) {
      try {
        await session.camera.stopStream();
        console.log(`🛑 Stream stopped`);
      } catch (err) {
        console.error(`Error stopping stream:`, err);
      }
      this.active.delete(userId);
      this.stopping.delete(userId);
    }

    try {
      sessions.remove(userId);
    } catch (err) {
      console.error(`Error during session cleanup for ${userId}:`, err);
    }
  }
}
