# 二创修改声明

本项目基于 `anjanatiha/Retrieval-Augmented-Generation-RAG-Agent` 二次开发。
原项目版权归原作者及贡献者所有，并继续遵循仓库中的 Apache License 2.0。

## 2026-09-26

- 将产品定位由通用 RAG 问答工具调整为个人文献阅读器，并命名为“知阅”。
- 重构 Streamlit 入口与中文导航，采用简约中性色视觉方案。
- 增加 SQLite 文献库及本地文件快照。
- 增加 PDF、DOCX、网页链接统一导入与重复检测。
- 增加 PDF/Word/网页原文预览和 AI 双栏阅读。
- 增加引用定位、阅读状态、进度、标签、收藏、笔记与删除功能。
- 修复空文献目录导致 Web UI 首次启动退出的问题。
- 增加个人文献库、预览模块及新版 UI 的自动化测试。

主要新增模块：

- `src/library/repository.py`
- `src/library/preview.py`
- `src/ui/literature.py`
- `tests/test_library_repository.py`
- `tests/test_library_preview.py`

主要修改文件：

- `app.py`
- `src/ui/handlers.py`
- `src/ui/theme.py`
- `src/rag/vector_store.py`
- `.streamlit/config.toml`

## 2026-09-27

- 按课程项目范围移除当前应用中的 AI 问答、向量索引、检索引用与 Ollama 初始化。
- 文献导入改为只解析并保存正文，不再建立向量索引。
- 阅读空间改为宽屏单栏，保留 PDF/Word/网页阅读、高亮、笔记、收藏与进度功能。
- 原项目 RAG 源码仍保留用于遵循二创溯源，但不再由“知阅”应用入口调用。
