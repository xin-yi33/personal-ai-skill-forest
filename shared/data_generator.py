"""
Data Generator for Skill Forest Experiments
Generates synthetic ToolBench-like API data across 5 domains.
"""
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# 5 domains as per the paper
DOMAINS = {
    "文档创作": {
        "keywords": ["文档", "word", "doc", "写作", "编辑", "排版", "文本", "text", "write", "edit", "letter", "essay", "report", "article"],
        "subcategories": ["word_processing", "text_editing", "document_formatting", "report_generation", "letter_writing", "article_writing"]
    },
    "数据分析": {
        "keywords": ["数据", "图表", "分析", "统计", "可视化", "chart", "data", "analysis", "statistics", "visualization", "graph", "plot", "dashboard", "excel", "csv"],
        "subcategories": ["chart_creation", "data_cleaning", "statistical_analysis", "visualization", "dashboard", "data_import"]
    },
    "通信协作": {
        "keywords": ["邮件", "消息", "日程", "协作", "会议", "通知", "email", "message", "calendar", "collaboration", "meeting", "notification", "chat", "schedule"],
        "subcategories": ["email_management", "messaging", "calendar", "meeting", "notification", "team_collaboration"]
    },
    "代码工程": {
        "keywords": ["代码", "编程", "调试", "部署", "API", "数据库", "code", "programming", "debug", "deploy", "database", "git", "testing", "build"],
        "subcategories": ["code_generation", "debugging", "deployment", "database", "api_development", "testing"]
    },
    "设计创意": {
        "keywords": ["图像", "设计", "视频", "动画", "海报", "Logo", "image", "design", "video", "animation", "poster", "creative", "illustration", "graphic"],
        "subcategories": ["image_generation", "graphic_design", "video_editing", "animation", "logo_design", "poster_creation"]
    }
}

# Skill templates per domain
SKILL_TEMPLATES = {
    "文档创作": [
        ("创建{format}文档", "创建一个新的{format}格式文档，支持文本输入和基本排版"),
        ("编辑{type}内容", "对文档中的{type}内容进行编辑修改，支持富文本格式"),
        ("文档格式转换", "将文档从{src_format}格式转换为{dst_format}格式"),
        ("生成{style}报告", "根据数据自动生成{style}风格的分析报告"),
        ("添加文档{element}元素", "在文档中添加{element}，支持自定义样式"),
        ("文档模板管理", "管理文档模板库，支持创建、编辑和应用模板"),
        ("文档版本控制", "跟踪文档修改历史，支持版本回退和对比"),
        ("批量文档处理", "批量处理多个文档，支持查找替换和格式统一"),
        ("文档协作编辑", "支持多人同时编辑文档，实现实时协作"),
        ("文档导出分享", "将文档导出为多种格式并生成分享链接"),
        ("文字排版优化", "优化文档排版，调整间距、字体和段落格式"),
        ("文档摘要生成", "自动生成文档的摘要和关键要点提取"),
        ("多语言文档翻译", "将文档内容翻译为指定语言"),
        ("文档水印添加", "为文档添加文字或图片水印"),
        ("文档目录生成", "根据标题自动生成文档目录结构"),
        ("智能文档校对", "检查文档中的拼写、语法和格式错误"),
        ("文档对比分析", "对比两个文档的差异并生成修改建议"),
        ("批量邮件合并", "将文档模板与收件人列表合并生成个性化邮件"),
        ("文档权限管理", "设置文档的查看、编辑和分享权限"),
        ("文档注释批注", "在文档中添加注释和批注，支持回复和讨论"),
    ],
    "数据分析": [
        ("创建{chart_type}图表", "根据数据创建{chart_type}，支持自定义样式和颜色"),
        ("数据清洗处理", "对原始数据进行清洗，处理缺失值、异常值和重复数据"),
        ("统计分析报告", "对数据进行描述性统计分析并生成报告"),
        ("数据透视表", "创建数据透视表进行多维度数据分析"),
        ("趋势分析预测", "分析数据趋势并生成预测模型"),
        ("数据可视化看板", "创建交互式数据可视化看板"),
        ("相关性分析", "分析变量之间的相关性并生成矩阵图"),
        ("数据导入导出", "支持多种格式的数据导入和导出"),
        ("时间序列分析", "对时间序列数据进行分析和预测"),
        ("聚类分析", "使用聚类算法对数据进行分组分析"),
        ("A/B测试分析", "对A/B测试结果进行统计分析"),
        ("数据采样处理", "对大数据集进行采样处理"),
        ("异常值检测", "自动检测数据中的异常值和离群点"),
        ("数据合并处理", "合并多个数据源的数据"),
        ("数据分组统计", "按指定维度对数据进行分组统计"),
        ("散点图回归分析", "创建散点图并添加回归分析线"),
        ("热力图生成", "生成数据热力图用于矩阵分析"),
        ("数据质量评估", "评估数据的完整性和质量"),
        ("自动报表生成", "根据模板自动生成数据分析报表"),
        ("数据对比分析", "对比不同时间段或组别的数据差异"),
    ],
    "通信协作": [
        ("发送{type}邮件", "发送{type}类型的邮件，支持附件和模板"),
        ("日程安排管理", "创建和管理日程安排，支持提醒功能"),
        ("会议通知发送", "向团队成员发送会议通知和议程"),
        ("消息群发", "向多个收件人群发消息，支持个性化内容"),
        ("邮件模板管理", "创建和管理邮件模板"),
        ("日历同步", "同步多个日历源的事件"),
        ("会议记录生成", "自动生成会议记录和行动项"),
        ("任务分配跟踪", "分配任务并跟踪完成进度"),
        ("团队公告发布", "向团队发布重要公告和通知"),
        ("邮件自动回复", "设置邮件自动回复规则"),
        ("通讯录管理", "管理联系人信息和分组"),
        ("消息通知设置", "配置消息通知的方式和频率"),
        ("视频会议安排", "安排和管理视频会议"),
        ("工作流审批", "创建工作流审批流程"),
        ("团队看板管理", "使用看板管理团队任务"),
        ("邮件签名管理", "创建和管理邮件签名"),
        ("定期报告汇总", "定期汇总团队工作进展"),
        ("协作空间创建", "创建团队协作空间"),
        ("消息搜索归档", "搜索和归档历史消息"),
        ("跨时区协调", "协调不同时区的会议安排"),
    ],
    "代码工程": [
        ("创建{lang}项目", "创建{lang}语言的项目框架，包含基本结构"),
        ("API接口开发", "开发RESTful API接口，支持认证和验证"),
        ("数据库设计", "设计数据库表结构和关系"),
        ("代码调试修复", "调试和修复代码中的错误"),
        ("单元测试编写", "编写单元测试用例，确保代码质量"),
        ("代码重构优化", "重构代码以提高可读性和性能"),
        ("CI/CD流水线配置", "配置持续集成和部署流水线"),
        ("代码审查工具", "使用工具进行代码审查和质量检查"),
        ("Docker容器化", "将应用容器化部署"),
        ("Git版本管理", "管理代码版本和分支策略"),
        ("API文档生成", "自动生成API接口文档"),
        ("性能优化分析", "分析和优化代码性能"),
        ("安全漏洞扫描", "扫描代码中的安全漏洞"),
        ("依赖管理", "管理项目依赖包"),
        ("日志系统配置", "配置应用日志系统"),
        ("缓存策略实现", "实现应用缓存策略"),
        ("消息队列集成", "集成消息队列系统"),
        ("微服务架构设计", "设计微服务架构方案"),
        ("代码生成工具", "使用代码生成工具提高效率"),
        ("部署脚本编写", "编写自动化部署脚本"),
    ],
    "设计创意": [
        ("生成{style}风格图像", "使用AI生成{style}风格的图像"),
        ("Logo设计", "设计品牌Logo，支持多种风格"),
        ("海报制作", "制作宣传海报，支持自定义模板"),
        ("图片编辑处理", "对图片进行裁剪、调色和滤镜处理"),
        ("动画效果制作", "制作动画效果，支持帧动画和补间动画"),
        ("UI界面设计", "设计用户界面，支持组件库"),
        ("图标素材生成", "生成图标素材，支持矢量格式"),
        ("背景图片生成", "生成各种风格的背景图片"),
        ("产品展示图", "制作产品展示图和效果图"),
        ("社交媒体图片", "为社交媒体平台制作图片"),
        ("名片设计", "设计个人或企业名片"),
        ("配色方案生成", "生成专业的配色方案"),
        ("插画创作", "创作各种风格的插画"),
        ("信息图制作", "制作信息图表"),
        ("视频封面设计", "为视频设计封面图"),
        ("品牌视觉设计", "设计品牌视觉识别系统"),
        ("网页原型设计", "设计网页原型图"),
        ("印刷品设计", "设计印刷品如传单、手册"),
        ("3D效果图", "生成3D效果图"),
        ("创意排版设计", "进行创意排版和版面设计"),
    ]
}


def generate_api(api_id: int, domain: str, subcategory: str, template: Tuple[str, str], idx: int) -> Dict:
    """Generate a single synthetic API entry."""
    name_template, desc_template = template
    # Fill in template variables
    fillers = {
        "{format}": np.random.choice(["Word", "PDF", "Markdown", "HTML", "RTF"]),
        "{type}": np.random.choice(["表格", "图片", "文字", "链接", "标题"]),
        "{src_format}": np.random.choice(["PDF", "Word", "TXT", "HTML"]),
        "{dst_format}": np.random.choice(["PDF", "Word", "Markdown", "HTML"]),
        "{style}": np.random.choice(["专业", "简约", "商务", "学术", "创意"]),
        "{element}": np.random.choice(["目录", "页眉", "页脚", "水印", "批注"]),
        "{chart_type}": np.random.choice(["柱状图", "折线图", "饼图", "散点图", "热力图"]),
        "{lang}": np.random.choice(["Python", "JavaScript", "Java", "Go", "Rust"]),
    }
    name = name_template
    desc = desc_template
    for key, val in fillers.items():
        name = name.replace(key, val)
        desc = desc.replace(key, val)

    name = f"{name}_{idx:03d}"

    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "输入数据"},
            "output_format": {"type": "string", "description": "输出格式"},
            "options": {"type": "object", "description": "附加选项"}
        }
    }

    return {
        "id": f"{domain}_{subcategory}_{idx:03d}",
        "name": name,
        "description": desc,
        "domain": domain,
        "subcategory": subcategory,
        "parameters": parameters,
        "requires": [],
        "version": "1.0.0"
    }


def generate_dataset(n_per_domain: int = 300, seed: int = 42) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """Generate a full synthetic dataset with n_per_domain APIs per domain."""
    np.random.seed(seed)
    all_domains = {}
    all_apis = []

    for domain, info in DOMAINS.items():
        templates = SKILL_TEMPLATES[domain]
        subcats = info["subcategories"]
        domain_apis = []
        for i in range(n_per_domain):
            template = templates[i % len(templates)]
            subcat = subcats[i % len(subcats)]
            api = generate_api(i, domain, subcat, template, i)
            domain_apis.append(api)
            all_apis.append(api)
        all_domains[domain] = domain_apis

    return all_domains, all_apis


def generate_test_queries(n_clear: int = 140, n_ambiguous: int = 60, seed: int = 42) -> List[Dict]:
    """Generate test queries: 70% clear intent + 30% cross-domain ambiguous."""
    np.random.seed(seed)
    queries = []

    # Base clear queries (100)
    clear_base = [
        ("帮我写一份项目策划书", "文档创作"), ("创建一个Word文档模板", "文档创作"),
        ("帮我格式化这份文档", "文档创作"), ("生成一份月度工作报告", "文档创作"),
        ("给文档添加目录", "文档创作"), ("把PDF转成Word格式", "文档创作"),
        ("帮我校对这份文档", "文档创作"), ("写一封正式的商务邮件", "文档创作"),
        ("文档排版优化一下", "文档创作"), ("帮我翻译这份文档", "文档创作"),
        ("添加文档水印", "文档创作"), ("批量处理这些文档", "文档创作"),
        ("生成文档摘要", "文档创作"), ("创建文档协作空间", "文档创作"),
        ("帮我写技术规范文档", "文档创作"), ("把文档导出成PDF", "文档创作"),
        ("帮我做个多语言文档", "文档创作"), ("文档版本对比", "文档创作"),
        ("帮我添加批注", "文档创作"), ("批量邮件合并", "文档创作"),
        ("帮我做一个柱状图展示销售数据", "数据分析"), ("分析这组数据的趋势", "数据分析"),
        ("做一个饼图展示市场份额", "数据分析"), ("帮我做个数据透视表", "数据分析"),
        ("把Excel数据可视化", "数据分析"), ("分析数据的相关性", "数据分析"),
        ("做一个交互式数据看板", "数据分析"), ("检测数据中的异常值", "数据分析"),
        ("对数据进行聚类分析", "数据分析"), ("生成统计分析报告", "数据分析"),
        ("做时间序列预测分析", "数据分析"), ("创建热力图展示矩阵数据", "数据分析"),
        ("合并多个Excel表格数据", "数据分析"), ("做A/B测试结果分析", "数据分析"),
        ("帮我做个散点图回归分析", "数据分析"), ("做数据质量评估", "数据分析"),
        ("生成自动报表", "数据分析"), ("数据对比分析", "数据分析"),
        ("帮我做分组统计", "数据分析"), ("做数据采样处理", "数据分析"),
        ("发送一封邮件给团队成员", "通信协作"), ("安排下周的团队会议", "通信协作"),
        ("给所有人发会议通知", "通信协作"), ("设置邮件自动回复", "通信协作"),
        ("创建团队协作空间", "通信协作"), ("生成会议纪要", "通信协作"),
        ("分配任务给团队成员", "通信协作"), ("发布团队公告", "通信协作"),
        ("同步多个日历", "通信协作"), ("管理通讯录", "通信协作"),
        ("设置消息通知规则", "通信协作"), ("创建工作流审批", "通信协作"),
        ("搜索历史消息记录", "通信协作"), ("安排跨时区会议", "通信协作"),
        ("创建团队看板", "通信协作"), ("邮件签名管理", "通信协作"),
        ("定期汇总报告", "通信协作"), ("消息搜索归档", "通信协作"),
        ("视频会议安排", "通信协作"), ("协作空间权限设置", "通信协作"),
        ("写一个Python爬虫程序", "代码工程"), ("开发一个用户登录API", "代码工程"),
        ("设计数据库表结构", "代码工程"), ("帮我调试这个代码错误", "代码工程"),
        ("编写单元测试用例", "代码工程"), ("配置CI/CD流水线", "代码工程"),
        ("把应用容器化部署", "代码工程"), ("生成API接口文档", "代码工程"),
        ("优化代码性能", "代码工程"), ("扫描代码安全漏洞", "代码工程"),
        ("创建一个React项目", "代码工程"), ("配置日志系统", "代码工程"),
        ("实现缓存策略", "代码工程"), ("设计微服务架构", "代码工程"),
        ("写个RESTful API", "代码工程"), ("帮我做代码重构", "代码工程"),
        ("配置消息队列", "代码工程"), ("编写部署脚本", "代码工程"),
        ("做代码审查", "代码工程"), ("管理项目依赖", "代码工程"),
        ("生成一张山水画风格的图片", "设计创意"), ("设计一个公司Logo", "设计创意"),
        ("制作一张宣传海报", "设计创意"), ("给图片加一个滤镜效果", "设计创意"),
        ("制作一个动画效果", "设计创意"), ("设计APP的UI界面", "设计创意"),
        ("生成一套图标素材", "设计创意"), ("设计一张名片", "设计创意"),
        ("制作信息图表", "设计创意"), ("设计社交媒体封面图", "设计创意"),
        ("创建品牌视觉方案", "设计创意"), ("设计网页原型图", "设计创意"),
        ("生成产品展示效果图", "设计创意"), ("做创意排版设计", "设计创意"),
        ("设计宣传册", "设计创意"), ("做3D效果图", "设计创意"),
        ("设计视频封面", "设计创意"), ("做印刷品设计", "设计创意"),
        ("生成背景图片", "设计创意"), ("设计配色方案", "设计创意"),
    ]

    ambiguous_base = [
        ("帮我处理一下这个文件", "文档创作"), ("帮我做个展示", "数据分析"),
        ("帮我分析一下这个API的文档", "代码工程"), ("帮我做个图表", "数据分析"),
        ("帮我写个自动化脚本", "代码工程"), ("帮我做个模板", "文档创作"),
        ("帮我处理这些数据", "数据分析"), ("帮我设计一个封面", "设计创意"),
        ("帮我写个函数", "代码工程"), ("帮我做个报告", "文档创作"),
        ("帮我做个界面", "设计创意"), ("帮我管理这个项目", "代码工程"),
        ("帮我发个通知", "通信协作"), ("帮我做个统计", "数据分析"),
        ("帮我写个文档", "文档创作"), ("帮我生成一张图片", "设计创意"),
        ("帮我安排一下", "通信协作"), ("帮我优化一下", "代码工程"),
        ("帮我创建一个看板", "数据分析"), ("帮我发一封邮件", "通信协作"),
        ("帮我做个动画", "设计创意"), ("帮我写个测试", "代码工程"),
        ("帮我整理一下通讯录", "通信协作"), ("帮我做个数据可视化", "数据分析"),
        ("帮我设计个海报", "设计创意"), ("帮我写个API", "代码工程"),
        ("帮我排版一下", "文档创作"), ("帮我发个群消息", "通信协作"),
        ("帮我画个图", "数据分析"), ("帮我做个Logo", "设计创意"),
        ("帮我做个数据报告", "数据分析"), ("帮我写技术文档", "文档创作"),
        ("帮我做个网页设计", "设计创意"), ("帮我设置日程提醒", "通信协作"),
        ("帮我写数据库查询", "代码工程"), ("帮我做个会议记录", "通信协作"),
        ("帮我处理图片", "设计创意"), ("帮我做个数据分析", "数据分析"),
        ("帮我写个项目计划", "文档创作"), ("帮我设计个UI", "设计创意"),
        ("帮我搞个表格", "数据分析"), ("帮我弄个脚本", "代码工程"),
        ("帮我做个幻灯片", "设计创意"), ("帮我弄个邮件模板", "通信协作"),
        ("帮我搞个数据看板", "数据分析"), ("帮我做个代码生成器", "代码工程"),
        ("帮我弄个通知系统", "通信协作"), ("帮我做个文件管理", "文档创作"),
        ("帮我搞个图表展示", "数据分析"), ("帮我做个视觉设计", "设计创意"),
    ]

    # Use as many as requested, cycling if needed
    for i in range(n_clear):
        q, domain = clear_base[i % len(clear_base)]
        queries.append({
            "query": q + (f" (v{i//len(clear_base)+1})" if i >= len(clear_base) else ""),
            "correct_domain": domain,
            "cross_domain_ambiguous": False,
            "correct_skill": f"{domain}_skill_{np.random.randint(0, 299):03d}"
        })

    for i in range(n_ambiguous):
        q, domain = ambiguous_base[i % len(ambiguous_base)]
        queries.append({
            "query": q + (f" (v{i//len(ambiguous_base)+1})" if i >= len(ambiguous_base) else ""),
            "correct_domain": domain,
            "cross_domain_ambiguous": True,
            "correct_skill": f"{domain}_skill_{np.random.randint(0, 299):03d}"
        })

    np.random.shuffle(queries)
    return queries

def compute_embeddings(apis: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """Compute embeddings for all API descriptions."""
    descriptions = [api['description'] for api in apis]
    embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)
    for api, emb in zip(apis, embeddings):
        api['embedding'] = emb
    return np.array(embeddings)


def save_dataset(all_domains: Dict, all_apis: List[Dict], test_queries: List[Dict], output_dir: str):
    """Save dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)

    apis_save = []
    for api in all_apis:
        api_copy = {k: v for k, v in api.items() if k != 'embedding'}
        if 'embedding' in api:
            api_copy['embedding'] = api['embedding'].tolist()
        apis_save.append(api_copy)

    with open(os.path.join(output_dir, 'all_apis.json'), 'w', encoding='utf-8') as f:
        json.dump(apis_save, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'test_queries.json'), 'w', encoding='utf-8') as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)

    domain_stats = {d: len(apis) for d, apis in all_domains.items()}
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(domain_stats, f, ensure_ascii=False, indent=2)


def load_dataset(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load dataset from disk."""
    with open(os.path.join(data_dir, 'all_apis.json'), 'r', encoding='utf-8') as f:
        all_apis = json.load(f)
    for api in all_apis:
        if 'embedding' in api:
            api['embedding'] = np.array(api['embedding'])

    with open(os.path.join(data_dir, 'test_queries.json'), 'r', encoding='utf-8') as f:
        test_queries = json.load(f)

    return all_apis, test_queries

