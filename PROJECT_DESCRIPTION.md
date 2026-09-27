# 知阅——个人文献阅读器项目说明

## 一、项目简介

“知阅”是一款面向个人学习与论文阅读场景的本地文献阅读器。项目基于开源项目
`Retrieval-Augmented-Generation-RAG-Agent` 进行二次开发，保留 Apache License 2.0
许可证和原作者署名。

本项目支持导入 PDF、Word（DOCX）和公开网页链接，并提供统一的文献管理、正文阅读、
划选高亮和阅读记录功能。当前版本取消了 AI 问答与向量检索，运行时不需要 Ollama。

## 二、主要功能

- PDF、Word、网页链接统一导入
- 文献标题、作者、标签和阅读状态管理
- 按标题、作者或标签搜索文献
- PDF 文本阅读与原版预览切换
- Word 和网页正文阅读
- 鼠标划选高亮、撤销上一次高亮、全部清除
- 阅读进度、收藏和个人笔记
- 本地 SQLite 数据存储与重复文献检测

## 三、技术实现

- 开发语言：Python 3.11
- Web 框架：Streamlit
- 数据存储：SQLite
- PDF 解析：PyMuPDF
- Word 解析：python-docx
- 网页解析：Requests、BeautifulSoup、lxml

文献文件保存在本机 `docs/` 目录，阅读记录保存在 `data/literature_library.db`。这些个人
数据不会提交到 Git 仓库。

## 四、运行方法

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

启动后在浏览器访问 `http://localhost:8501/`。

## 五、二创说明

本次二创主要增加了中文简约界面、个人文献库、三类文献统一导入、宽屏阅读、高亮、
阅读进度、收藏和笔记等功能。详细修改记录见 `MODIFICATIONS.md`。

