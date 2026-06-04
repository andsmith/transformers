import { defineConfig } from "vite";
import { resolve } from "node:path";

// Served as a GitHub Pages project page at `/transformers/`. Building with this
// base means every emitted asset URL resolves under the subpath — both at
// andsmith.github.io/transformers/ today and andsmith.net/transformers/ later
// (see README "Hosting" for the apex migration that flips the second URL on).
export default defineConfig({
  base: "/transformers/",
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
      },
    },
  },
});
