from bs4 import BeautifulSoup

def extract_text_from_html(html: str) -> str:

    """ Helper - mirrors what scraper_agent does internally"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def test_extracts_paragraph_text():
    html="<html><body><p>Quantum Computing is Powerful.</p></body></html>"
    result=extract_text_from_html(html)
    assert "Quantum Computing is Powerful." in result

def test_removes_script_tags():
    html="<html><body><p>Real content</p><script>alert('noise')</script></body></html>"
    result=extract_text_from_html(html)
    assert "alert" not in result
    assert "Real content" in result

def test_removes_nav_tags():
    html = "<html><body><nav>Home | About</nav><p>Article content</p></body></html>"
    result=extract_text_from_html(html)
    assert "Home | About" not in result
    assert "Article content" in result

def test_empty_body_returns_empty():
    html = "<html><body></body></html>"
    result=extract_text_from_html(html)
    assert result==""