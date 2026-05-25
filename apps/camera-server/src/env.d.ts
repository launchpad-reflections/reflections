declare namespace NodeJS {
  interface ProcessEnv {
    PACKAGE_NAME: string;
    MENTRAOS_API_KEY: string;
    PORT?: string;
    WHIP_URL?: string;
  }
}
