import { execSync } from "node:child_process";

function fail(message) {
  console.error(`\n❌ ${message}\n`);
  process.exit(1);
}

// 0. Skip if GITHUB_TOKEN is already set in azd environment
try {
  const existing = execSync("azd env get-value GITHUB_TOKEN", {
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "ignore"],
  }).trim();
  if (existing) {
    console.log("✓ GITHUB_TOKEN already set in azd environment.");
    process.exit(0);
  }
} catch {
  // Not set yet — continue to retrieve it
}

// 1. Check gh CLI is installed
try {
  execSync("gh --version", { stdio: "ignore" });
} catch {
  fail(
    "GitHub CLI (gh) is not installed.\n" +
    "   This project requires gh to obtain a GitHub token for the Copilot SDK.\n" +
    "   Install it from: https://cli.github.com"
  );
}

// 2. Check gh CLI is authenticated
let token;
try {
  token = execSync("gh auth token", { encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] }).trim();
} catch {
  fail(
    "GitHub CLI is not authenticated.\n" +
    "   Run: gh auth login\n" +
    "   Then re-run: azd up"
  );
}

// 3. Verify copilot scope is present (warn but don't fail)
try {
  const scopes = execSync("gh auth status", { encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"] });
  const combined = scopes || "";
  if (!combined.includes("copilot")) {
    // Also check stderr output where gh auth status writes
    try {
      const stderr = execSync("gh auth status 2>&1", { encoding: "utf-8", shell: true });
      if (!stderr.includes("copilot")) {
        console.warn(
          "⚠ The 'copilot' token scope may be missing.\n" +
          "   If the agent fails to authenticate, run: gh auth refresh --scopes copilot"
        );
      }
    } catch {
      // Best-effort scope check — proceed with token we have
    }
  }
} catch {
  // gh auth status can fail in some CI/hook contexts — proceed with token from step 2
  console.warn("⚠ Could not verify token scopes. Proceeding with available token.");
}

// 4. Persist token to azd environment
// Token is passed via env var to avoid exposing it in process argument lists
const isWindows = process.platform === "win32";
const cmd = isWindows
  ? `azd env set GITHUB_TOKEN %__GH_TOKEN%`
  : `azd env set GITHUB_TOKEN "$__GH_TOKEN"`;
execSync(cmd, {
  env: { ...process.env, __GH_TOKEN: token },
  stdio: "inherit",
  shell: true,
});
console.log("✓ GITHUB_TOKEN set from gh CLI.");
