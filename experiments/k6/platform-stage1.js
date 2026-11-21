// k6 load test for the Platform API at stage-1 rates (ticket P9, ekido-ai/ekido-internal#66).
//
// Stage 1 is 1,000 teams. The scaling design (14-scaling-design.md §3.7) puts stage 2 (100,000 teams) at "tens of
// requests per second on average, hundreds at peak". Scaled by 1/100 that is ~0.3 req/s average and ~3 req/s at peak
// for stage 1 (an estimate). This test runs ~6 req/s in total, twice the stage-1 peak, for 5 minutes.
//
// Endpoints, one scenario each, with a constant arrival rate:
//   sign-in      POST /v1/auth/email/verify with a wrong token: the sign-in code path, without sending email
//   me           GET  /v1/me                          (session lookup, every signed-in page)
//   team         GET  /v1/orgs/{org}                  (Settings: team and members)
//   keys         GET  /v1/orgs/{org}/api-keys         (Settings: API keys)
//   quotes       POST /v1/orgs/{org}/quotes           (a fresh quote for one dataset made in setup)
//   recipes      GET  /v1/recipes                     (public)
//   jobs         GET  /sepolia/jobs?limit=40          (the job list the dashboard and admin read; the platform has no
//                                                      customer job list yet)
//
// Env: BASE (default staging), SESSION (ekido_session cookie of a test account), ORG (that account's team id),
//      OUT (where to write the JSON summary), DURATION (default 5m).
import http from "k6/http";
import { check } from "k6";

const BASE = __ENV.BASE || "https://ekido-staging.duckdns.org";
const API = `${BASE}/sepolia/v1`;
const ORG = __ENV.ORG;
const DURATION = __ENV.DURATION || "5m";
const params = (name, extra = {}) => ({
  headers: { cookie: `ekido_session=${__ENV.SESSION}`, "content-type": "application/json" },
  tags: { name },
  ...extra,
});

const RATES = { signin: 0.5, me: 1, team: 1, keys: 1, quotes: 0.5, recipes: 1, jobs: 1 }; // requests per second
const scenario = (exec, perSecond) => ({
  executor: "constant-arrival-rate",
  exec,
  rate: Math.round(perSecond * 10),
  timeUnit: "10s",
  duration: DURATION,
  preAllocatedVUs: 4,
  maxVUs: 20,
});

export const options = {
  scenarios: Object.fromEntries(Object.entries(RATES).map(([name, r]) => [name, scenario(name, r)])),
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max", "count"],
  // A threshold per endpoint makes k6 report each one's numbers separately. They are set high on purpose:
  // this run records what the API does, it doesn't gate anything.
  thresholds: Object.fromEntries(
    Object.keys(RATES).flatMap((name) => [
      [`http_req_duration{name:${name}}`, ["p(95)<60000"]],
      [`http_req_failed{name:${name}}`, ["rate<1"]],
    ]),
  ),
};

export function setup() {
  if (!__ENV.SESSION || !ORG) throw new Error("set SESSION and ORG");
  // One dataset for the quote scenario: the smallest valid ticket-sorter set (1 train row, 20 test rows).
  const labels = ["complaint", "order", "question", "refund", "other"];
  const test = Array.from({ length: 20 }, (_, i) => ({ text: `k6 load test ticket ${i + 1}: order kab aayega?`, label: labels[i % labels.length] }));
  const body = JSON.stringify({ recipe_id: "ticket-sorter-hinglish", train: [{ text: "k6 train row: refund chahiye", label: "refund" }], test });
  const res = http.post(`${API}/orgs/${ORG}/datasets`, body, params("setup"));
  if (res.status !== 201) throw new Error(`dataset setup failed: ${res.status} ${res.body}`);
  return { datasetId: res.json("dataset_id") };
}

export function signin() {
  // A wrong token runs the sign-in lookup and must be refused, so 4xx is the expected answer here.
  const res = http.post(`${API}/auth/email/verify`, JSON.stringify({ token: "k6-not-a-real-token-000000000000000000000" }),
    params("signin", { responseCallback: http.expectedStatuses({ min: 400, max: 499 }) }));
  check(res, { "signin refused a wrong token": (r) => r.status >= 400 && r.status < 500 });
}
export function me() {
  check(http.get(`${API}/me`, params("me")), { "me 200": (r) => r.status === 200 });
}
export function team() {
  check(http.get(`${API}/orgs/${ORG}`, params("team")), { "team 200": (r) => r.status === 200 });
}
export function keys() {
  check(http.get(`${API}/orgs/${ORG}/api-keys`, params("keys")), { "keys 200": (r) => r.status === 200 });
}
export function quotes(data) {
  const res = http.post(`${API}/orgs/${ORG}/quotes`, JSON.stringify({ dataset_id: data.datasetId }), params("quotes"));
  check(res, { "quote 201": (r) => r.status === 201 });
}
export function recipes() {
  check(http.get(`${API}/recipes`, { tags: { name: "recipes" } }), { "recipes 200": (r) => r.status === 200 });
}
export function jobs() {
  check(http.get(`${BASE}/sepolia/jobs?limit=40`, { tags: { name: "jobs" } }), { "jobs 200": (r) => r.status === 200 });
}

export function handleSummary(data) {
  const per = {};
  for (const name of Object.keys(RATES)) {
    const d = data.metrics[`http_req_duration{name:${name}}`]?.values ?? {};
    const f = data.metrics[`http_req_failed{name:${name}}`]?.values ?? {};
    per[name] = {
      requests: d.count ?? 0,
      target_per_second: RATES[name],
      p50_ms: d.med, p90_ms: d["p(90)"], p95_ms: d["p(95)"], p99_ms: d["p(99)"], max_ms: d.max,
      error_rate: f.rate,
    };
  }
  const out = {
    test: "platform-stage1",
    ticket: "ekido-ai/ekido-internal#66",
    base: BASE,
    started_at: __ENV.STARTED_AT || null,
    duration: DURATION,
    total_target_per_second: Object.values(RATES).reduce((a, b) => a + b, 0),
    machine: __ENV.MACHINE || null,
    k6_version: __ENV.K6_VERSION || null,
    endpoints: per,
    checks: Object.fromEntries(Object.entries(data.root_group.checks ?? {}).map(([k, c]) => [c.name ?? k, { passes: c.passes, fails: c.fails }])),
  };
  const result = { stdout: `${JSON.stringify(out.endpoints, null, 1)}\n` };
  if (__ENV.OUT) result[__ENV.OUT] = JSON.stringify(out, null, 2);
  return result;
}
