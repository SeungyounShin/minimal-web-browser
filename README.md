## Minimal Web Browser Env

`minimal-web-browser-env`는 튜토리얼용으로 만든 초경량 브라우저 환경입니다.  
기본적으로는 결정적인 synthetic 페이지를 제공하지만, `open id=https://...` 형태로 실제 URL을 건네면
해당 페이지를 가져와 Markdown 텍스트와 링크 목록을 즉시 만들어 줍니다. `select` 액션을 사용하면 현재
페이지의 링크를 id 또는 라벨 일부로 따라갈 수 있으며, 같은 문서 내 앵커를 클릭하면 자동으로 해당 구간으로
스크롤된 상태로 표시됩니다.

### 사전 준비

이 저장소는 [uv](https://github.com/astral-sh/uv)를 기준으로 구성했습니다.

```bash
cd /Users/robin/Desktop/seungyoun/projects/minimal-web-browser-env
uv sync
```

### CLI 실행

```bash
uv run minimal-browser
```

#### DuckDuckGo 백엔드 및 출력 행 수 제어

환경 변수를 설정해 백엔드와 기본 출력 행 수를 변경할 수 있습니다.

```bash
MINI_BROWSER_BACKEND=duckduckgo MINI_BROWSER_VIEW_LINES=60 uv run minimal-browser
```

명령어 예시는 다음과 같습니다.

google scholar style search  
(DuckDuckGo 결과 HTML에서 `HTML (experimental)` 링크를 찾고 해당 URL로 이동합니다.)
```
action> search query="Attention Is All You Need Google Scholar"
action> open id=2                 # DuckDuckGo → arXiv abstract
action> open id=24                # HTML (experimental) 링크
action> open id=7                 # Scaled Dot-Product Attention 섹션
action> find pattern="d_{k}"      # dk\sqrt{d_{k}} 부분 강조 (옵션)
action> open id=7                 # 전체 본문으로 복귀
action> snapshot max_lines=220 include_links=false
action> quit
```

각 액션이 실행될 때마다 현재 커서, 페이지 미리보기, 사용 가능한 링크 목록을 함께 출력합니다.

논문 서치 예시
(`MINI_BROWSER_BACKEND=duckduckgo` 환경 변수를 켠 상태라면 DuckDuckGo Lite 실시간 검색 페이지가 열립니다. 그렇지 않으면 튜토리얼 데이터가 반환됩니다.)
```
action> search query="Tongyi Lab Publications"
action> open id=4
action> find pattern=attention
```

### 파이썬에서 사용하기

```python
import asyncio
from minimal_web_browser_env import WebBrowserEnv, DuckDuckGoBackend

async def demo():
    env = WebBrowserEnv(
        backend=DuckDuckGoBackend(),
        preview_lines=60,   # 기본 출력 라인 수
    )
    await env.reset()
    await env.step("search", query="Attention Is All You Need")
    await env.step("open", id=2)   # arXiv abstract
    await env.step("open", id=24)  # HTML (experimental)
    state = await env.step("snapshot", max_lines=150, include_links=False)
    print(state["current_page"]["title"])

asyncio.run(demo())
```
