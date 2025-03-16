from contextlib import asynccontextmanager
import asyncio
import logging
import random
import re
import json
from pathlib import Path
from typing import Dict, Tuple, List, cast
from urllib.parse import urlparse
import os
from bs4 import BeautifulSoup
import bs4

class Browser:
    logger = logging.getLogger(__name__)
    max_browsers = 3
    browser_load_threshold = 5
    browsers: set["Browser"] = set()
    browsers_lock = asyncio.Lock()

    @staticmethod
    def get_domain(url: str) -> str:
        domain = urlparse(url).netloc
        parts = domain.split(".")
        if len(parts) > 2:
            domain = ".".join(parts[-2:])
        return domain

    def __init__(self, driver=None):
        self.driver = driver
        self.processing_count = 0
        self.has_blank_page = True
        self.domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.tab_mode = True
        self.max_scroll_percent = 500
        self.stopping = False
        self._cookies = None

    @property
    def cookies(self):
        """쿠키 관리를 위한 CookieJar 인스턴스를 반환합니다."""
        if self._cookies is None and self.driver is not None:
            self._cookies = CookieJar(self.driver, self.logger)
        return self._cookies

    async def get_driver(self, headless: bool = False):
        try:
            global zendriver
            import zendriver as zd
        except ImportError:
            raise ImportError(
                "The zendriver package is required to use Browser. "
                "Please install it with: pip install zendriver"
            )

        config = zd.Config(
            headless=headless,
            browser_connection_timeout=3,
        )
        driver = await zd.start(config)
        return driver

    async def login(self, login_url: str, username: str, password: str):
        """로그인 페이지에 접속하여 로그인합니다."""
        self.logger.info(f"로그인 시도 중: {login_url}")
        
        page = await self.get(login_url)
        
        try:
            # 페이지가 완전히 로드될 때까지 대기
            await page.wait(3)
            
            id= await page.select('input[name="id"]')
            await id.send_keys(username)
            pw= await page.select('input[name="password"]')
            await pw.send_keys(password)

            # 폼 제출
            await page.evaluate('''
                (() => {
                    // 로그인 버튼 클릭
                    const loginButton = document.querySelector('input.btnLogin[type="submit"]');
                    if (loginButton) {
                        loginButton.click();
                        return true;
                    }
                    
                    // 버튼을 찾지 못한 경우 폼 제출 시도
                    const form = document.getElementById('login');
                    if (form) {
                        form.submit();
                        return true;
                    }
                    
                    return false;
                })()
            ''')

            # 네트워크 요청이 완료될 때까지 대기
            await page.wait(3)
            
                        
            # 로그인 성공 여부 확인
            current_url = await page.evaluate('window.location.href')
            self.logger.info(f"현재 URL: {current_url}")
            
            if "login" not in current_url:
                self.logger.info("로그인 성공")
                await self.cookies.save()
            else:
                self.logger.warning("로그인 후에도 로그인 페이지에 있습니다. 로그인이 실패했을 수 있습니다.")
            
            return page
        except Exception as e:
            self.logger.error(f"로그인 실패: {str(e)}")
            if page:
                await self.close_page(page)
            raise

    async def check_login_status(self, home_url: str = 'https://medline.inje.ac.kr/login'):
        """현재 로그인 상태인지 확인합니다."""
        self.logger.info("로그인 상태 확인 중...")
        
        # 로그인 페이지 URL 저장
        login_url = home_url
        
        try:
            # 먼저, 쿠키 로드 시도
            await self.cookies.load()
            self.logger.info("저장된 세션 쿠키를 로드했습니다.")
            
            # 로그인 페이지로 이동
            page = await self.get(login_url)
            
            # 페이지 로드 대기
            await page.wait(2)
            
            # 현재 URL 가져오기
            current_url = await page.evaluate('window.location.href')
            self.logger.info(f"초기 URL: {login_url}")
            self.logger.info(f"현재 URL: {current_url}")
            
            # 리다이렉트 확인: 로그인 페이지로 접속했는데 메인 페이지로 리다이렉트되면 로그인된 상태
            is_redirected = current_url != login_url
            is_main_page = "login" not in current_url
            
            # 리다이렉트되었고 메인 페이지면 로그인 상태
            is_logged_in = is_redirected and is_main_page
            
            # 추가 확인: 리다이렉트되지 않은 경우에도 로그인 폼이 없으면 로그인된 상태일 수 있음
            if not is_logged_in:
                login_form_exists = await page.evaluate('''
                    (() => {
                        return !!document.getElementById('login') || 
                               !!document.querySelector('input[name="id"]') ||
                               !!document.querySelector('form[action*="login"]');
                    })()
                ''')
                
                # 로그인 폼이 없으면 로그인된 상태
                is_logged_in = not login_form_exists
                
                self.logger.info(f"로그인 폼 존재 여부: {login_form_exists}")
            
            self.logger.info(f"리다이렉트 여부: {is_redirected}, 메인 페이지 여부: {is_main_page}")
            self.logger.info(f"최종 로그인 상태 판단: {is_logged_in}")
            
            if is_logged_in:
                self.logger.info("로그인 상태입니다.")
            else:
                self.logger.info("로그인 상태가 아닙니다.")
                
            return is_logged_in
            
        except Exception as e:
            self.logger.error(f"로그인 상태 확인 실패: {str(e)}")
            if 'page' in locals() and page:
                await self.close_page(page)
            return False

    async def get(self, url: str):
        """URL로 이동하여 페이지를 가져옵니다."""
        self.processing_count += 1
        try:
            async with self.rate_limit_for_domain(url):
                new_window = not self.has_blank_page
                self.has_blank_page = False
                try:
                    return await self.driver.get(url)
                except asyncio.CancelledError:
                    self.logger.warning(f"페이지 로드 중 작업이 취소되었습니다: {url}")
                    raise
                except Exception as e:
                    self.logger.error(f"페이지 로드 실패: {url}, 오류: {str(e)}")
                    raise
        except Exception:
            self.processing_count -= 1
            raise

    async def scroll_page_to_bottom(self, page):
        """페이지를 바닥까지 스크롤합니다."""
        total_scroll_percent = 0
        while True:
            # 탭 모드에서는 스크롤하기 전에 탭을 전면으로 가져옵니다
            if self.tab_mode:
                await page.bring_to_front()
            scroll_percent = random.randrange(46, 97)
            total_scroll_percent += scroll_percent
            await page.scroll_down(scroll_percent)
            await page.wait(2)
            await page.sleep(random.uniform(0.23, 0.56))

            if total_scroll_percent >= self.max_scroll_percent:
                break

            # 페이지 끝에 도달했는지 확인
            is_at_bottom = await page.evaluate(
                "window.innerHeight + window.scrollY >= document.scrollingElement.scrollHeight"
            )
            if is_at_bottom:
                break

    @asynccontextmanager
    async def rate_limit_for_domain(self, url: str):
        """도메인별 속도 제한을 적용합니다."""
        semaphore = None
        try:
            domain = self.get_domain(url)

            semaphore = self.domain_semaphores.get(domain)
            if not semaphore:
                semaphore = asyncio.Semaphore(1)
                self.domain_semaphores[domain] = semaphore

            was_locked = semaphore.locked()
            async with semaphore:
                if was_locked:
                    await asyncio.sleep(random.uniform(0.6, 1.2))
                yield

        except Exception as e:
            # 오류를 기록하지만 요청을 차단하지 않습니다
            self.logger.warning(f"Rate limiting error for {url}: {str(e)}")
            yield

    async def scrape_page_with_doi(self, doi: str):
        """DOI를 사용하여 페이지를 스크래핑합니다."""
        url = f"https://doi-org-ssl.mproxy.inje.ac.kr/{doi}"
        self.logger.info(f"DOI 스크래핑 시작: {doi}")
        
        page = await self.get(url)
        
        try:
            await page.wait(3)
            # HTML 콘텐츠 가져오기
            html = await page.get_content()
            
            # 페이지 제목 가져오기
            title = await page.evaluate('document.title')
            safe_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
            
            # BeautifulSoup으로 HTML 파싱
            soup = BeautifulSoup(html, 'html.parser')
            
            # 스크래핑 후 HTML 정리
            soup = self.clean_soup(soup)
            
            self.logger.info(f"DOI 스크래핑 완료: {doi}, 제목: {safe_title}")
            return safe_title, soup
        except Exception as e:
            self.logger.error(f"DOI 스크래핑 실패: {doi}, 오류: {str(e)}")
            raise

    def clean_soup(self, soup: BeautifulSoup) -> BeautifulSoup:
        """HTML에서 불필요한 요소를 제거하여 정리합니다"""
        # 불필요한 태그 제거
        for tag in soup.find_all(
            [
                "script",
                "style",
                "footer",
                "header",
                "nav",
                "menu",
                "sidebar",
                "svg",
            ]
        ):
            tag.decompose()

        # 특정 클래스를 가진 태그 제거
        disallowed_class_set = {"nav", "menu", "sidebar", "footer", "advertisement", "ad", "banner"}

        # 특정 클래스를 가진 태그 확인 함수
        def does_tag_have_disallowed_class(elem) -> bool:
            if not isinstance(elem, bs4.Tag):
                return False

            return any(
                cls_name in disallowed_class_set for cls_name in elem.get("class", [])
            )

        # 특정 클래스를 가진 태그 제거
        for tag in soup.find_all(does_tag_have_disallowed_class):
            tag.decompose()
            
        # 특정 ID를 가진 태그 제거
        disallowed_id_set = {"nav", "menu", "sidebar", "footer", "header", "advertisement", "ad", "banner"}
        for tag in soup.find_all(id=lambda x: x and x.lower() in disallowed_id_set):
            tag.decompose()
            
        # 빈 태그 제거
        for tag in soup.find_all():
            if tag.name != 'br' and not tag.contents and not tag.string:
                tag.decompose()
        
        # 불필요한 공백 제거
        # 1. 모든 텍스트 노드에서 연속된 공백을 하나로 줄이기
        for text in soup.find_all(text=True):
            if text.parent.name not in ['script', 'style', 'pre', 'code']:
                # 연속된 공백을 하나로 줄이기
                new_text = re.sub(r'\s+', ' ', text.string.strip())
                text.replace_with(new_text)
        
        # 2. 빈 텍스트 노드 제거
        for text in soup.find_all(text=True):
            if not text.strip():
                text.extract()
        
        # 3. 줄바꿈 정리 - 연속된 줄바꿈을 하나로 줄이기
        html_content = str(soup)
        html_content = re.sub(r'\n\s*\n', '\n', html_content)
        
        # 수정된 HTML로 새 BeautifulSoup 객체 생성
        soup = BeautifulSoup(html_content, 'html.parser')

        return soup

    async def close_page(self, page):
        """페이지를 닫습니다."""
        try:
            await page.close()
        finally:
            self.processing_count -= 1

    async def stop(self):
        """브라우저를 중지합니다."""
        if self.stopping:
            return
        self.stopping = True
        await self.driver.stop()

    @classmethod
    async def get_browser(cls, headless: bool = False) -> "Browser":
        """사용 가능한 브라우저를 가져오거나 새 브라우저를 생성합니다."""
        async with cls.browsers_lock:
            if len(cls.browsers) == 0:
                # 사용 가능한 브라우저가 없으면 새로 생성
                return await cls.create(headless)

            # 부하 분산: 탭 수가 가장 적은 브라우저 선택
            browser = min(cls.browsers, key=lambda b: b.processing_count)

            # 모든 브라우저가 과부하 상태이고 더 생성할 수 있는 경우
            if (
                browser.processing_count >= cls.browser_load_threshold
                and len(cls.browsers) < cls.max_browsers
            ):
                return await cls.create(headless)

            return browser

    @classmethod
    async def release_browser(cls, browser: "Browser"):
        """브라우저를 해제합니다."""
        async with cls.browsers_lock:
            if browser and browser.processing_count <= 0:
                try:
                    await browser.stop()
                finally:
                    cls.browsers.discard(browser)

    async def load_storage_state(self, page, storage_path):
        """저장된 브라우저 상태를 로드합니다."""
        try:
            storage_path = Path(storage_path)
            if storage_path.exists():
                self.logger.info(f"저장된 세션 데이터 로드 중: {storage_path}")
                with open(storage_path, "r") as f:
                    storage_state = json.load(f)
                
                # 세션 저장 방식에 따라 적절한 메서드 호출
                if hasattr(self.driver, "load_storage_state"):
                    await self.driver.load_storage_state(storage_state)
                    self.logger.info("브라우저 세션 데이터 로드 완료")
                else:
                    # 쿠키만 로드하는 대체 방법
                    self.logger.warning("load_storage_state 메서드가 없어 쿠키만 로드합니다.")
                    if "cookies" in storage_state and hasattr(self.driver, "cookies"):
                        await self.driver.cookies.load(storage_state["cookies"])
                        self.logger.info("쿠키 데이터 로드 완료")
                    else:
                        self.logger.warning("쿠키 데이터를 로드할 수 없습니다.")
                
                return True
            else:
                self.logger.warning(f"세션 파일이 존재하지 않습니다: {storage_path}")
                return False
        except Exception as e:
            self.logger.error(f"세션 데이터 로드 실패: {str(e)}")
            return False

    async def save_storage_state(self, page, storage_path):
        """현재 브라우저 상태를 저장합니다."""
        try:
            self.logger.info(f"세션 데이터 저장 중: {storage_path}")
            storage_state = {}
            
            # 세션 저장 방식에 따라 적절한 메서드 호출
            if hasattr(self.driver, "storage_state"):
                storage_state = await self.driver.storage_state()
            elif hasattr(self.driver, "cookies"):
                # 쿠키만 저장하는 대체 방법
                cookies = await self.driver.cookies.get_all()
                storage_state = {"cookies": cookies}
            else:
                self.logger.warning("세션 데이터를 저장할 방법이 없습니다.")
                return False
            
            storage_path = Path(storage_path)
            os.makedirs(storage_path.parent, exist_ok=True)
            
            with open(storage_path, "w") as f:
                json.dump(storage_state, f)
            
            self.logger.info(f"세션 데이터 저장 완료: {storage_path}")
            return True
        except Exception as e:
            self.logger.error(f"세션 데이터 저장 실패: {str(e)}")
            return False

class CookieJar:
    def __init__(self, driver, logger=None):
        self.driver = driver
        self.logger = logger or logging.getLogger(__name__)
        self.default_file = '.session.dat'

    async def save(self, filepath=None):
        """쿠키를 파일에 저장합니다. 경로를 지정하지 않으면 .session.dat에 저장됩니다."""
        filepath = filepath or self.default_file
        try:
            self.logger.info(f"쿠키 데이터 저장 중: {filepath}")
            
            if hasattr(self.driver, "cookies") and hasattr(self.driver.cookies, "save"):
                # zendriver의 자체 save 메서드 사용 (파일 경로 지정 가능한 경우)
                await self.driver.cookies.save(filepath)
                self.logger.info(f"쿠키 데이터 저장 완료: {filepath}")
                return True
            elif hasattr(self.driver, "storage_state"):
                # storage_state 메서드를 통해 쿠키 저장
                storage_state = await self.driver.storage_state()
                
                filepath_path = Path(filepath)
                os.makedirs(filepath_path.parent, exist_ok=True)
                
                with open(filepath_path, "w") as f:
                    json.dump(storage_state, f)
                
                self.logger.info(f"쿠키 데이터 저장 완료: {filepath}")
                return True
            else:
                self.logger.warning("쿠키 데이터를 저장할 방법이 없습니다.")
                return False
        except Exception as e:
            self.logger.error(f"쿠키 데이터 저장 실패: {str(e)}")
            return False

    async def load(self, filepath=None):
        """파일에서 쿠키를 로드합니다. 경로를 지정하지 않으면 .session.dat에서 로드합니다."""
        filepath = filepath or self.default_file
        try:
            filepath_path = Path(filepath)
            if filepath_path.exists():
                self.logger.info(f"쿠키 데이터 로드 중: {filepath}")
                
                if hasattr(self.driver, "cookies") and hasattr(self.driver.cookies, "load"):
                    # zendriver의 자체 load 메서드 사용
                    await self.driver.cookies.load(filepath)
                    self.logger.info("쿠키 데이터 로드 완료")
                    return True
                else:
                    # 수동으로 쿠키 로드
                    with open(filepath_path, "r") as f:
                        storage_state = json.load(f)
                    
                    if hasattr(self.driver, "load_storage_state"):
                        await self.driver.load_storage_state(storage_state)
                        self.logger.info("쿠키 데이터 로드 완료")
                    elif "cookies" in storage_state and hasattr(self.driver, "cookies"):
                        await self.driver.cookies.load(storage_state["cookies"])
                        self.logger.info("쿠키 데이터 로드 완료")
                    else:
                        self.logger.warning("쿠키 데이터를 로드할 수 없습니다.")
                        return False
                    
                    return True
            else:
                self.logger.warning(f"쿠키 파일이 존재하지 않습니다: {filepath}")
                return False
        except Exception as e:
            self.logger.error(f"쿠키 데이터 로드 실패: {str(e)}")
            return False

    async def get_all(self, requests_cookie_format=False):
        """모든 쿠키를 가져옵니다. requests_cookie_format=True이면 requests 라이브러리와 호환되는 형식으로 반환합니다."""
        try:
            if hasattr(self.driver, "cookies") and hasattr(self.driver.cookies, "get_all"):
                cookies = await self.driver.cookies.get_all()
                
                if requests_cookie_format and cookies:
                    # requests 라이브러리와 호환되는 형식으로 변환
                    try:
                        import requests.cookies
                        converted_cookies = []
                        for cookie in cookies:
                            cookie_obj = requests.cookies.create_cookie(
                                name=cookie.get('name'),
                                value=cookie.get('value'),
                                domain=cookie.get('domain'),
                                path=cookie.get('path'),
                                expires=cookie.get('expires'),
                                secure=cookie.get('secure', False),
                                rest={'HttpOnly': cookie.get('httpOnly', False)}
                            )
                            converted_cookies.append(cookie_obj)
                        return converted_cookies
                    except ImportError:
                        self.logger.warning("requests 라이브러리가 설치되어 있지 않아 변환할 수 없습니다.")
                        return cookies
                
                return cookies
            elif hasattr(self.driver, "storage_state"):
                # storage_state에서 쿠키 추출
                storage = await self.driver.storage_state()
                if "cookies" in storage:
                    cookies = storage["cookies"]
                    
                    if requests_cookie_format and cookies:
                        # requests 형식으로 변환 (위와 동일)
                        try:
                            import requests.cookies
                            converted_cookies = []
                            for cookie in cookies:
                                cookie_obj = requests.cookies.create_cookie(
                                    name=cookie.get('name'),
                                    value=cookie.get('value'),
                                    domain=cookie.get('domain'),
                                    path=cookie.get('path'),
                                    expires=cookie.get('expires'),
                                    secure=cookie.get('secure', False),
                                    rest={'HttpOnly': cookie.get('httpOnly', False)}
                                )
                                converted_cookies.append(cookie_obj)
                            return converted_cookies
                        except ImportError:
                            self.logger.warning("requests 라이브러리가 설치되어 있지 않아 변환할 수 없습니다.")
                            return cookies
                    
                    return cookies
            
            self.logger.warning("쿠키를 가져올 수 없습니다.")
            return []
        except Exception as e:
            self.logger.error(f"쿠키 가져오기 실패: {str(e)}")
            return []

async def initialize_browser(username: str = "108514", password: str = "gs12341234!") -> Browser:
    """
    브라우저를 초기화하고 필요한 경우에만 로그인합니다.
    
    Args:
        username: 로그인 사용자 이름
        password: 로그인 비밀번호
        
    Returns:
        Browser: 초기화된 브라우저 인스턴스
    """
    # 먼저 드라이버를 생성합니다
    try:
        global zendriver
        import zendriver as zd
    except ImportError:
        raise ImportError(
            "The zendriver package is required to use Browser. "
            "Please install it with: pip install zendriver"
        )

    config = zd.Config(
        headless=False,
        browser_connection_timeout=3,
    )
    driver = await zd.start(config)
    
    # 브라우저 인스턴스 생성
    browser = Browser(driver=driver)

    Browser.logger.info("브라우저가 생성되었습니다.")
    Browser.logger.info("로그인 여부를 확인 중")
    is_logged_in = await browser.check_login_status()
    if is_logged_in:
        Browser.logger.info("이미 로그인되어 있습니다.")
    else:
        Browser.logger.info("로그인이 필요합니다. 로그인 시도 중...")
        login_url = "https://medline.inje.ac.kr/login"
        await browser.login(login_url, username, password)
    
    return browser

