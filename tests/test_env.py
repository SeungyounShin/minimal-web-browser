import asyncio

from minimal_web_browser_env import WebBrowserEnv


def test_env_cycle():
    async def runner():
        env = WebBrowserEnv()
        state = await env.reset()
        assert state["history"] == []

        state = await env.step("search", query="demo")
        assert len(state["history"]) == 1
        assert state["links"], "search results should expose links"

        state = await env.step("open", id=0)
        assert "gpt-oss" in state["current_page"]["title"].lower()
        assert "browser actions" not in state["last_response"][0].splitlines()[0].lower()

        state = await env.step("open", id=0)
        assert "browser actions" in state["last_response"][0].lower()

        state = await env.step("select", target="tutorial")
        assert "gpt-oss" in state["current_page"]["title"].lower()

        snap = await env.step("snapshot")
        assert "snapshot" in snap["last_response"][0].lower()

        state = await env.step("find", pattern="browser")
        assert "match at" in state["last_response"][0].lower()

    asyncio.run(runner())

