import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

export default defineConfig({
  site: "https://nvidia.github.io",
  base: "/NVFlare",
  integrations: [tailwind()],
});
