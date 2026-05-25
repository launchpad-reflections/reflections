import "dotenv/config";
import { CameraApp } from "./server/CameraApp";

const packageName = process.env.PACKAGE_NAME;
const apiKey = process.env.MENTRAOS_API_KEY;
const port = Number(process.env.PORT ?? 3000);

if (!packageName || !apiKey) {
  throw new Error("PACKAGE_NAME and MENTRAOS_API_KEY must be set in .env");
}

const app = new CameraApp({ packageName, apiKey, port });
await app.start();
console.log(`✅ Listening on :${port}`);

const shutdown = async () => {
  try { await app.stop(); } catch {}
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);