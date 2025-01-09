import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

const branch = process.env.PUBLIC_GH_BRANCH;

export default defineConfig({
  site: "https://nvidia.github.io",
  base: branch === 'main' ? '/NVFlare' : `/NVFlare/version/${branch}`,
  integrations: [tailwind()],
});
