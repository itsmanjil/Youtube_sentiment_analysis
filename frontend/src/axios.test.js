import { afterEach, describe, expect, test, vi } from "vitest";

const { getStoredAccessTokenMock, requestSilentRefreshMock } = vi.hoisted(() => ({
  getStoredAccessTokenMock: vi.fn(),
  requestSilentRefreshMock: vi.fn(),
}));

vi.mock("./utils/auth", () => ({
  getStoredAccessToken: getStoredAccessTokenMock,
  requestSilentRefresh: requestSilentRefreshMock,
}));

const makeResponse = (config, status) => ({
  data: { detail: status === 401 ? "unauthorized" : "ok" },
  status,
  statusText: status === 401 ? "Unauthorized" : "OK",
  headers: {},
  config,
});

describe("axiosInstance response interceptor (401 -> silent refresh -> retry)", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  const loadAxiosInstance = async (adapter) => {
    const mod = await import("./axios");
    const axiosInstance = mod.default;
    axiosInstance.defaults.adapter = adapter;
    return axiosInstance;
  };

  test("retries once with a fresh token after a successful silent refresh", async () => {
    getStoredAccessTokenMock
      .mockReturnValueOnce("stale-token")
      .mockReturnValueOnce("fresh-token");
    requestSilentRefreshMock.mockResolvedValueOnce(true);

    let calls = 0;
    const adapter = vi.fn(async (config) => {
      calls += 1;
      return makeResponse(config, calls === 1 ? 401 : 200);
    });

    const axiosInstance = await loadAxiosInstance(adapter);
    const response = await axiosInstance.get("youtube/analyses/");

    expect(adapter).toHaveBeenCalledTimes(2);
    expect(requestSilentRefreshMock).toHaveBeenCalledTimes(1);
    expect(response.status).toBe(200);
    expect(adapter.mock.calls[1][0].headers.Authorization).toBe(
      "Bearer fresh-token"
    );
  });

  test("does not loop if the retried request still 401s", async () => {
    getStoredAccessTokenMock.mockReturnValue("stale-token");
    requestSilentRefreshMock.mockResolvedValueOnce(true);

    const adapter = vi.fn(async (config) => makeResponse(config, 401));
    const axiosInstance = await loadAxiosInstance(adapter);
    const response = await axiosInstance.get("youtube/analyses/");

    expect(adapter).toHaveBeenCalledTimes(2);
    expect(response.status).toBe(401);
  });

  test("returns the original 401 without retrying if the refresh fails", async () => {
    getStoredAccessTokenMock.mockReturnValue("stale-token");
    requestSilentRefreshMock.mockResolvedValueOnce(false);

    const adapter = vi.fn(async (config) => makeResponse(config, 401));
    const axiosInstance = await loadAxiosInstance(adapter);
    const response = await axiosInstance.get("youtube/analyses/");

    expect(adapter).toHaveBeenCalledTimes(1);
    expect(response.status).toBe(401);
  });

  test("does not attempt a refresh for the login/refresh endpoints themselves", async () => {
    getStoredAccessTokenMock.mockReturnValue(null);

    const adapter = vi.fn(async (config) => makeResponse(config, 401));
    const axiosInstance = await loadAxiosInstance(adapter);

    const refreshResponse = await axiosInstance.post("token/refresh/", {});
    const loginResponse = await axiosInstance.post("token/", {});

    expect(requestSilentRefreshMock).not.toHaveBeenCalled();
    expect(adapter).toHaveBeenCalledTimes(2);
    expect(refreshResponse.status).toBe(401);
    expect(loginResponse.status).toBe(401);
  });
});
