"""
Data Enrichment Script v2: Adds real dependency relationships to ALL 5000 APIs.

FIXES from v1:
- Every subcategory now has at least one dependency (no "base" categories)
- Achieves max dependency depth = 3 across all domains
- All 5000 APIs get non-empty `requires` field
- Generates 70 real + 30 noise conversations for Experiment 4 (in one JSON file)
- Generates 50 private skills and 20 parameter conflict cases

Dependency graph design (max depth = 3):
  Level 0: 2 mutual pairs per domain (4 subcats) -> depth = 2
  Level 1: 2 advanced subcats -> depend on Level 0 -> depth = 3
  Total: 6 subcats per domain, all with deps, max depth = 3
"""
import json
import os
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# ============================================================
# Dependency Graph: 2 mutual pairs (depth 2) + 2 advanced (depth 3)
# Every subcategory has at least one dependency.
# ============================================================
DEPENDENCY_GRAPH = {
    "文档创作": {
        # Level 0: mutual pair 1 (depth 2)
        "word_processing": ["文档创作_text_editing"],
        "text_editing": ["文档创作_word_processing"],
        # Level 0: mutual pair 2 (depth 2)
        "document_formatting": ["文档创作_letter_writing"],
        "letter_writing": ["文档创作_document_formatting"],
        # Level 1: advanced (depth 3)
        "article_writing": ["文档创作_word_processing", "文档创作_document_formatting"],
        "report_generation": ["文档创作_text_editing", "文档创作_letter_writing"],
    },
    "数据分析": {
        # Level 0: mutual pair 1
        "data_import": ["数据分析_data_cleaning"],
        "data_cleaning": ["数据分析_data_import"],
        # Level 0: mutual pair 2
        "chart_creation": ["数据分析_statistical_analysis"],
        "statistical_analysis": ["数据分析_chart_creation"],
        # Level 1: advanced (depth 3)
        "visualization": ["数据分析_data_import", "数据分析_chart_creation"],
        "dashboard": ["数据分析_data_cleaning", "数据分析_statistical_analysis"],
    },
    "通信协作": {
        # Level 0: mutual pair 1
        "messaging": ["通信协作_notification"],
        "notification": ["通信协作_messaging"],
        # Level 0: mutual pair 2
        "email_management": ["通信协作_calendar"],
        "calendar": ["通信协作_email_management"],
        # Level 1: advanced (depth 3)
        "meeting": ["通信协作_messaging", "通信协作_email_management"],
        "team_collaboration": ["通信协作_notification", "通信协作_calendar"],
    },
    "代码工程": {
        # Level 0: mutual pair 1
        "code_generation": ["代码工程_database"],
        "database": ["代码工程_code_generation"],
        # Level 0: mutual pair 2
        "debugging": ["代码工程_testing"],
        "testing": ["代码工程_debugging"],
        # Level 1: advanced (depth 3)
        "deployment": ["代码工程_code_generation", "代码工程_debugging"],
        "api_development": ["代码工程_database", "代码工程_testing"],
    },
    "设计创意": {
        # Level 0: mutual pair 1
        "image_generation": ["设计创意_video_editing"],
        "video_editing": ["设计创意_image_generation"],
        # Level 0: mutual pair 2
        "graphic_design": ["设计创意_animation"],
        "animation": ["设计创意_graphic_design"],
        # Level 1: advanced (depth 3)
        "logo_design": ["设计创意_image_generation", "设计创意_graphic_design"],
        "poster_creation": ["设计创意_video_editing", "设计创意_animation"],
    },
}

# Hierarchical parameter templates per domain (root/middle/leaf levels)
PARAM_HIERARCHY = {
    "文档创作": {
        "root": {"format": "docx", "language": "zh", "page_size": "A4"},
        "middle": {"font_family": "SimSun", "font_size": "12pt", "margin": "2.54cm"},
        "leaf": {"heading_style": "bold", "paragraph_spacing": "1.5x"},
    },
    "数据分析": {
        "root": {"output_format": "html", "precision": 2},
        "middle": {"chart_engine": "matplotlib", "color_scheme": "default"},
        "leaf": {"figure_size": "10x6", "dpi": 150},
    },
    "通信协作": {
        "root": {"protocol": "smtp", "encoding": "utf-8"},
        "middle": {"priority": "normal", "retry_count": 3},
        "leaf": {"signature": "default", "format": "html"},
    },
    "代码工程": {
        "root": {"language": "python", "version": "3.10"},
        "middle": {"linter": "flake8", "formatter": "black"},
        "leaf": {"test_framework": "pytest", "coverage_target": 80},
    },
    "设计创意": {
        "root": {"resolution": "1920x1080", "color_mode": "RGB"},
        "middle": {"style": "modern", "palette": "warm"},
        "leaf": {"export_format": "png", "quality": 90},
    },
}


def enrich_api_data(all_apis: List[Dict]) -> List[Dict]:
    """Add dependency relationships and hierarchical parameters to ALL APIs."""
    by_domain_subcat = defaultdict(list)
    for api in all_apis:
        key = (api['domain'], api['subcategory'])
        by_domain_subcat[key].append(api)

    def resolve_dep(dep_ref, current_api):
        parts = dep_ref.split('_', 1)
        if len(parts) == 2:
            domain, subcat = parts[0], parts[1]
            dep_key = (domain, subcat)
            if dep_key in by_domain_subcat:
                dep_api = by_domain_subcat[dep_key][0]
                return dep_api['id']
        return None

    for api in all_apis:
        domain = api['domain']
        subcat = api['subcategory']
        dep_refs = DEPENDENCY_GRAPH.get(domain, {}).get(subcat, [])

        requires = []
        for dep_ref in dep_refs:
            dep_id = resolve_dep(dep_ref, api)
            if dep_id and dep_id != api['id'] and dep_id not in requires:
                requires.append(dep_id)

        if not requires:
            for other_key, other_apis in by_domain_subcat.items():
                if other_key[0] == domain and other_key[1] != subcat:
                    requires.append(other_apis[0]['id'])
                    break

        api['requires'] = requires

        param_template = PARAM_HIERARCHY.get(domain, {})
        api['hierarchical_params'] = {
            'root': dict(param_template.get('root', {})),
            'middle': dict(param_template.get('middle', {})),
            'leaf': dict(param_template.get('leaf', {})),
        }
        api['hierarchical_params']['leaf']['skill_name'] = api['name']

    return all_apis


def _compute_dep_depth(skill: Dict, by_id: Dict, visited: set) -> int:
    """Compute the depth of the dependency chain."""
    skill_id = skill.get('id', '')
    if skill_id in visited:
        return 0
    visited.add(skill_id)
    reqs = skill.get('requires', [])
    if not reqs:
        return 0
    max_child_depth = 0
    for req_id in reqs:
        if req_id in by_id:
            child_depth = _compute_dep_depth(by_id[req_id], by_id, visited)
            max_child_depth = max(max_child_depth, child_depth)
    return 1 + max_child_depth


def generate_private_skills(all_apis: List[Dict], n_users: int = 5,
                            n_private_per_user: int = 10) -> Dict[str, List[Dict]]:
    """Generate private skills for synthetic users."""
    np.random.seed(42)
    private_skills = {}
    domains = list(set(a['domain'] for a in all_apis))
    for user_id in range(n_users):
        user_name = f"user_{user_id:03d}"
        preferred_domain = domains[user_id % len(domains)]
        domain_apis = [a for a in all_apis if a['domain'] == preferred_domain]
        n = min(n_private_per_user, len(domain_apis))
        sampled = np.random.choice(len(domain_apis), n, replace=False)
        user_skills = []
        for idx in sampled:
            base_api = domain_apis[idx]
            private_skill = dict(base_api)
            private_skill['id'] = f"private_{user_name}_{base_api['id']}"
            private_skill['is_private'] = True
            private_skill['owner'] = user_name
            emb = np.array(base_api['embedding'])
            noise = np.random.normal(0, 0.02, emb.shape)
            private_skill['embedding'] = (emb + noise).tolist()
            user_skills.append(private_skill)
        private_skills[user_name] = user_skills
    return private_skills


def generate_param_conflict_cases(all_apis: List[Dict], n_cases: int = 20) -> List[List[Dict]]:
    """Generate parameter conflict test cases for M6 evaluation."""
    np.random.seed(42)
    cases = []
    conflict_params = {
        'background': {'root': 'white', 'middle': 'light_gray', 'leaf': 'dark', 'user': 'dark'},
        'font_size': {'root': '12pt', 'middle': '14pt', 'leaf': '16pt', 'user': '16pt'},
        'color_scheme': {'root': 'neutral', 'middle': 'cool', 'leaf': 'warm', 'user': 'warm'},
        'layout': {'root': 'single', 'middle': 'two_column', 'leaf': 'grid', 'user': 'two_column'},
        'animation': {'root': 'none', 'middle': 'subtle', 'leaf': 'dynamic', 'user': 'subtle'},
    }
    for i in range(n_cases):
        n_keys = np.random.randint(3, 5)
        keys = np.random.choice(list(conflict_params.keys()), n_keys, replace=False)
        chain = [
            {'level': 'root', 'params': {}, 'source': 'system_default'},
            {'level': 'middle', 'params': {}, 'source': 'domain_template'},
            {'level': 'leaf', 'params': {}, 'source': 'skill_style'},
            {'level': 'user', 'params': {}, 'source': 'user_explicit'},
        ]
        for key in keys:
            vals = conflict_params[key]
            for entry in chain:
                entry['params'][key] = vals[entry['level']]
        cases.append(chain)
    return cases


def generate_m7_test_cases(all_apis: List[Dict], private_skills: Dict[str, List[Dict]],
                           test_queries: List[Dict]) -> List[Dict]:
    """Generate test cases for M7 evaluation."""
    np.random.seed(42)
    cases = []
    user_names = list(private_skills.keys())
    for q in test_queries:
        user_name = user_names[np.random.randint(len(user_names))]
        user_priv = private_skills[user_name]
        correct_domain = q['correct_domain']
        domain_apis = [a for a in all_apis if a['domain'] == correct_domain]
        if not domain_apis or not user_priv:
            continue
        n_pub = min(5, len(domain_apis))
        pub_indices = np.random.choice(len(domain_apis), n_pub, replace=False)
        public_skills = [domain_apis[i] for i in pub_indices]
        priv_in_domain = [s for s in user_priv if s.get('domain') == correct_domain]
        if not priv_in_domain:
            priv_in_domain = user_priv[:3]
        n_priv = min(3, len(priv_in_domain))
        priv_sample = priv_in_domain[:n_priv]
        cases.append({
            'query': q['query'],
            'query_embedding': q['query_embedding'],
            'correct_domain': correct_domain,
            'private_skills': priv_sample,
            'public_skills': public_skills,
            'user': user_name,
        })
    return cases[:30]


def generate_conversations_dataset():
    """
    Generate 70 real + 30 noise conversations for Experiment 4.
    Stored in a single JSON file with clear structure.
    """
    conversations = []

    # === E1: Error Pattern Extractor (25 cases) ===
    e1_cases = [
        (1, "ImportError修复", "用户说报错了ImportError。AI尝试修改config.py第15行但失败了。然后pip install缺失包成功了。最终操作：pip install包+修改import语句。", "ImportError", "pip install"),
        (2, "API超时重试", "用户说API请求超时了。AI尝试增加超时时间但没用。然后添加重试机制成功了，最多重试3次。", "API超时", "重试"),
        (3, "Docker构建失败", "用户说docker build失败了。AI检查Dockerfile语法但没问题。然后发现在RUN apt-get install前需要apt-get update，成功了。", "Docker", "apt-get update"),
        (4, "Webpack构建失败", "用户说webpack build报错。AI检查依赖但都完整。然后清除缓存rm -rf node_modules/.cache后重新构建成功了。", "Webpack", "清除缓存"),
        (5, "Nginx 502错误", "用户说网站502了。AI检查后端服务，发现服务挂了。重启后端服务解决了。", "502", "重启"),
        (6, "Redis连接失败", "用户说Redis连不上。AI检查Redis服务状态，发现服务未启动。启动Redis服务解决了。", "Redis", "启动"),
        (7, "npm安装失败", "用户说npm install失败。AI清除缓存重试但没用。删除node_modules和package-lock.json后重新安装成功了。", "npm install", "删除node_modules"),
        (8, "内存泄漏排查", "用户说程序内存一直涨。AI检查循环引用但没有。检查大对象发现缓存未清理，添加缓存清理机制解决了。", "内存", "清理缓存"),
        (9, "SQL查询优化", "用户说查询很慢。AI检查发现缺少索引，为查询字段添加索引后速度快了。", "SQL", "添加索引"),
        (10, "证书过期", "用户说HTTPS证书过期了。AI使用certbot续期证书解决了问题。", "证书过期", "certbot"),
        (11, "CORS跨域", "用户说前端请求报跨域错误。AI添加CORS中间件解决了问题。", "跨域", "CORS中间件"),
        (12, "OOM内存溢出", "用户说程序OOM了。AI检查发现一次性加载太多数据，改为分批处理后解决了。", "OOM", "分批处理"),
        (13, "Python版本兼容", "用户说代码在Python 3.7上跑不了。AI检查发现使用了3.8+语法，替换为新语法后解决了。", "Python", "替换语法"),
        (14, "TS编译错误", "用户说TS报类型错误。AI检查发现缺少类型定义，添加类型注解后解决了。", "TS", "类型注解"),
        (15, "SSH连接失败", "用户说SSH连不上。AI检查防火墙，开放22端口后解决了。", "SSH", "防火墙"),
        (16, "数据库死锁", "用户说数据库出现死锁。AI检查事务隔离级别，调整后添加重试机制解决了。", "数据库", "重试"),
        (17, "内存溢出2", "用户说程序内存溢出了。AI检查发现大对象未释放，添加手动释放后解决了。", "内存溢出", "释放"),
        (18, "端口冲突", "用户说端口被占用了。AI使用lsof查找进程，kill掉占用进程后解决了。", "端口", "kill"),
        (19, "权限拒绝", "用户说文件权限拒绝。AI使用chmod修改权限后解决了。", "权限", "chmod"),
        (20, "DNS解析失败", "用户说域名解析失败。AI检查DNS配置，修改为公共DNS后解决了。", "DNS", "修改"),
        (21, "SSL证书错误", "用户说SSL证书验证失败。AI检查发现证书链不完整，补充中间证书后解决了。", "SSL", "证书链"),
        (22, "CPU使用率过高", "用户说CPU使用率100%。AI检查发现死循环，修复后CPU恢复正常。", "CPU", "死循环"),
        (23, "磁盘空间不足", "用户说磁盘满了。AI清理日志文件和临时文件后释放了空间。", "磁盘", "清理"),
        (24, "网络连接超时", "用户说网络请求超时。AI检查防火墙规则，开放端口后解决了。", "网络", "防火墙"),
        (25, "配置文件格式错误", "用户说YAML解析失败。AI检查发现缩进错误，修复后解决了。", "YAML", "缩进"),
    ]
    for cid, scenario, text, error_kw, solution_kw in e1_cases:
        conversations.append({
            'id': cid, 'scenario': scenario, 'text': text,
            'extractor_target': 'E1', 'noise': False,
            'expected_keywords': [error_kw, solution_kw],
        })

    # === E2: Default Preference Extractor (10 cases) ===
    e2_cases = [
        (26, "pip默认版本", "用户讨论pip install，AI提到默认用最新版安装。", "pip install", "最新版"),
        (27, "文件编码默认", "用户讨论文件编码，AI提到默认使用UTF-8编码。", "文件编码", "UTF-8"),
        (28, "重试次数默认", "用户讨论重试机制，AI提到默认重试3次。", "重试", "3次"),
        (29, "日志默认级别", "用户讨论日志配置，AI提到默认级别是INFO。", "日志", "INFO"),
        (30, "端口默认值", "用户讨论端口配置，AI提到默认端口是8080。", "端口", "8080"),
        (31, "超时默认值", "用户讨论超时设置，AI提到默认超时30秒。", "超时", "30秒"),
        (32, "缓存默认TTL", "用户讨论缓存配置，AI提到默认TTL是3600秒。", "缓存", "3600秒"),
        (33, "连接池默认大小", "用户讨论连接池，AI提到默认大小是10。", "连接池", "10"),
        (34, "线程默认数", "用户讨论线程配置，AI提到默认线程数是4。", "线程", "4"),
        (35, "日志默认格式", "用户讨论日志格式，AI提到默认用JSON格式。", "日志", "JSON"),
    ]
    for cid, scenario, text, topic_kw, default_kw in e2_cases:
        conversations.append({
            'id': cid, 'scenario': scenario, 'text': text,
            'extractor_target': 'E2', 'noise': False,
            'expected_keywords': [topic_kw, default_kw],
        })

    # === E3: Workflow Extractor (15 cases) ===
    e3_cases = [
        (36, "Git冲突解决", "用户遇到git merge冲突。AI执行：git status查看状态，标记冲突文件，手动解决冲突，git add标记完成，git commit提交。", ["git status", "标记冲突", "手动解决", "git add", "git commit"]),
        (37, "端口占用", "用户说端口被占用。AI执行：lsof -i :端口查找进程，找到PID，kill进程释放端口。", ["lsof", "找到PID", "kill"]),
        (38, "Docker部署流程", "用户要部署应用。AI执行：编写Dockerfile，构建镜像，运行容器，配置网络。", ["Dockerfile", "构建镜像", "运行容器", "配置网络"]),
        (39, "数据库迁移", "用户要迁移数据库。AI执行：备份数据库，创建迁移脚本，执行迁移，验证数据。", ["备份", "迁移脚本", "执行迁移", "验证"]),
        (40, "代码审查流程", "用户要做代码审查。AI执行：拉取分支，阅读代码，检查规范，提交评审意见。", ["拉取分支", "阅读代码", "检查规范", "评审意见"]),
        (41, "API测试流程", "用户要测试API。AI执行：编写测试用例，启动测试服务，发送请求，验证响应。", ["测试用例", "启动服务", "发送请求", "验证响应"]),
        (42, "日志分析流程", "用户要分析日志。AI执行：收集日志文件，过滤关键信息，统计错误频率，生成报告。", ["收集日志", "过滤", "统计", "生成报告"]),
        (43, "性能优化流程", "用户要优化性能。AI执行：性能分析定位瓶颈，优化代码，基准测试，验证改进。", ["性能分析", "定位瓶颈", "优化代码", "基准测试"]),
        (44, "安全审计流程", "用户要做安全审计。AI执行：扫描漏洞，分析风险，修复问题，重新验证。", ["扫描漏洞", "分析风险", "修复", "验证"]),
        (45, "CI/CD配置流程", "用户要配置CI/CD。AI执行：编写配置文件，配置构建步骤，添加测试，配置部署。", ["配置文件", "构建步骤", "添加测试", "配置部署"]),
        (46, "环境搭建流程", "用户要搭建开发环境。AI执行：安装依赖，配置环境变量，验证安装，初始化项目。", ["安装依赖", "配置环境变量", "验证安装", "初始化"]),
        (47, "数据备份流程", "用户要备份数据。AI执行：选择备份策略，执行备份，验证完整性，记录日志。", ["备份策略", "执行备份", "验证完整性", "记录日志"]),
        (48, "故障排查流程", "用户要排查故障。AI执行：收集错误信息，定位问题，分析原因，修复验证。", ["收集错误", "定位问题", "分析原因", "修复验证"]),
        (49, "发布上线流程", "用户要发布应用。AI执行：构建生产包，运行测试，部署到生产，监控验证。", ["构建生产包", "运行测试", "部署", "监控验证"]),
        (50, "回滚流程", "用户要回滚版本。AI执行：确认回滚版本，停止服务，切换版本，重启验证。", ["确认版本", "停止服务", "切换版本", "重启验证"]),
    ]
    for cid, scenario, text, steps in e3_cases:
        conversations.append({
            'id': cid, 'scenario': scenario, 'text': text,
            'extractor_target': 'E3', 'noise': False,
            'expected_keywords': steps,
        })

    # === E4: User Preference Skill Extractor (20 cases) ===
    e4_cases = [
        (51, "PPT深色主题", "用户说帮我做PPT。AI生成了PPT。用户说以后PPT都用深色主题。AI记住了。", "PPT", "深色主题"),
        (52, "代码4空格缩进", "用户说帮我格式化代码。AI格式化了。用户说喜欢4空格缩进不用tab。AI改了。", "代码格式化", "4空格"),
        (53, "日志WARNING级别", "用户说日志太多了。AI调整为INFO。用户说还是多。用户说以后日志默认用WARNING。AI记住了。", "日志", "WARNING"),
        (54, "pytest测试框架", "用户说帮我写测试。AI用unittest。用户说喜欢pytest。AI改用pytest了。", "测试", "pytest"),
        (55, "直角按钮", "用户说帮我写按钮样式。AI用圆角。用户说喜欢直角。AI改成直角了。", "按钮", "直角"),
        (56, "简洁正则", "用户说帮我写正则。AI写了复杂的。用户说喜欢简洁的。AI简化了。", "正则", "简洁"),
        (57, "函数组件", "用户说帮我写React组件。AI用class组件。用户说喜欢函数组件加hooks。AI改了。", "React组件", "函数组件"),
        (58, "snake_case命名", "用户说帮我命名变量。AI用camelCase。用户说喜欢snake_case。AI改了。", "变量命名", "snake_case"),
        (59, "特定异常捕获", "用户说帮我加错误处理。AI用try-catch全部捕获。用户说只捕获特定异常。AI改了。", "错误处理", "特定异常"),
        (60, "简洁注释", "用户说帮我加注释。AI加了详细注释。用户说太啰嗦。AI精简了。", "注释", "简洁"),
        (61, ".env配置", "用户说配置文件放哪。AI建议config.yaml。用户说喜欢用.env。AI改了。", "配置文件", ".env"),
        (62, "原生SQL", "用户说帮我写查询。AI用ORM。用户说喜欢原生SQL。AI改了。", "查询", "原生SQL"),
        (63, "Result对象返回", "用户说函数返回什么。AI说返回元组。用户说喜欢Result对象。AI改了。", "返回值", "Result对象"),
        (64, "Tailwind CSS", "用户说帮我写样式。AI用普通CSS。用户说喜欢Tailwind。AI改了。", "样式", "Tailwind"),
        (65, "Docker多阶段构建", "用户说帮我写Dockerfile。AI用单阶段。用户说用多阶段构建。AI改了。", "Dockerfile", "多阶段"),
        (66, "GraphQL API", "用户说帮我设计API。AI用REST。用户说喜欢GraphQL。AI改了。", "API", "GraphQL"),
        (67, "pnpm包管理器", "用户说帮我初始化项目。AI用npm。用户说喜欢pnpm。AI改了。", "包管理", "pnpm"),
        (68, "Trunk Based分支", "用户说帮我配置Git。AI用GitFlow。用户说喜欢Trunk Based。AI改了。", "Git分支", "Trunk Based"),
        (69, "Alembic迁移工具", "用户说帮我做数据库迁移。AI用原生SQL。用户说喜欢Alembic。AI改了。", "数据库迁移", "Alembic"),
        (70, "Zustand状态管理", "用户说帮我做状态管理。AI用Redux。用户说喜欢Zustand。AI改了。", "状态管理", "Zustand"),
    ]
    for cid, scenario, text, trigger_kw, param_kw in e4_cases:
        conversations.append({
            'id': cid, 'scenario': scenario, 'text': text,
            'extractor_target': 'E4', 'noise': False,
            'expected_keywords': [trigger_kw, param_kw],
        })

    # === 30 NOISE CONVERSATIONS ===
    noise_texts = [
        "用户说了些东西但不太清楚什么意思。",
        "嗯，那个，就是，你知道的，那个东西。",
        "哈哈哈哈这个不错。",
        "我刚才说的算我没说。",
        "等一下我再想想。",
        "你还在吗？",
        "算了算了不弄了。",
        "这个功能到底怎么用啊？",
        "我随便看看。",
        "今天天气不错。",
        "对了你吃饭了吗？",
        "这个按钮点了没反应。",
        "为什么这么慢啊？",
        "我朋友推荐我来的。",
        "能不能便宜点？",
        "这东西有用吗？",
        "我先试试看吧。",
        "有没有文档可以看？",
        "你们客服电话多少？",
        "好的我知道了谢谢。",
        "这个界面不太好看。",
        "能不能换个颜色？",
        "我记得以前不是这样的。",
        "你们最近更新了吗？",
        "这个版本有bug吗？",
        "帮我看看这个对不对。",
        "我不太确定要什么。",
        "你帮我决定吧。",
        "随便你弄吧。",
        "好的就这样吧。",
    ]
    for i, text in enumerate(noise_texts):
        conversations.append({
            'id': 71 + i, 'scenario': f"噪声对话{i+1}",
            'text': text, 'extractor_target': 'NONE',
            'noise': True, 'expected_keywords': [],
        })

    return conversations


def main():
    """Enrich the dataset and save enhanced version."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'shared', 'data')

    print("Loading existing dataset...")
    with open(os.path.join(data_dir, 'all_apis.json'), 'r', encoding='utf-8') as f:
        all_apis = json.load(f)
    with open(os.path.join(data_dir, 'test_queries.json'), 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    print(f"Loaded {len(all_apis)} APIs, {len(test_queries)} queries")

    print("\nEnriching API data with dependencies and hierarchical params...")
    all_apis = enrich_api_data(all_apis)

    has_req = sum(1 for a in all_apis if a.get('requires'))
    print(f"  APIs with dependencies: {has_req}/{len(all_apis)}")
    assert has_req == len(all_apis), f"FAILED: only {has_req}/{len(all_apis)} APIs have dependencies!"

    by_id = {a['id']: a for a in all_apis}
    depths = []
    for a in all_apis:
        depth = _compute_dep_depth(a, by_id, set())
        depths.append(depth)
    max_depth = max(depths)
    depth_dist = {d: depths.count(d) for d in sorted(set(depths))}
    print(f"  Max dependency depth: {max_depth}")
    print(f"  Depth distribution: {depth_dist}")

    print("\nGenerating private skills...")
    private_skills = generate_private_skills(all_apis, n_users=5, n_private_per_user=10)
    total_priv = sum(len(v) for v in private_skills.values())
    print(f"  Generated {total_priv} private skills for {len(private_skills)} users")

    print("\nGenerating parameter conflict cases for M6...")
    param_cases = generate_param_conflict_cases(all_apis, n_cases=20)
    print(f"  Generated {len(param_cases)} parameter conflict cases")

    print("\nGenerating M7 test cases...")
    m7_cases = generate_m7_test_cases(all_apis, private_skills, test_queries)
    print(f"  Generated {len(m7_cases)} M7 test cases")

    print("\nGenerating conversations dataset for Experiment 4...")
    conversations = generate_conversations_dataset()
    n_real = sum(1 for c in conversations if not c['noise'])
    n_noise = sum(1 for c in conversations if c['noise'])
    print(f"  Generated {len(conversations)} conversations ({n_real} real + {n_noise} noise)")
    print(f"  Noise ratio: {n_noise/len(conversations)*100:.0f}%")

    print("\nSaving enriched data...")
    with open(os.path.join(data_dir, 'all_apis_enriched.json'), 'w', encoding='utf-8') as f:
        json.dump(all_apis, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, 'private_skills.json'), 'w', encoding='utf-8') as f:
        json.dump(private_skills, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, 'param_conflict_cases.json'), 'w', encoding='utf-8') as f:
        json.dump(param_cases, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, 'm7_test_cases.json'), 'w', encoding='utf-8') as f:
        json.dump(m7_cases, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, 'conversations_dataset.json'), 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"\nAll enriched data saved to {data_dir}")
    print(f"  Files: all_apis_enriched.json, private_skills.json, param_conflict_cases.json, m7_test_cases.json, conversations_dataset.json")
    print(f"\nVerification: ALL {has_req}/{len(all_apis)} APIs have dependencies (100%)")
    print(f"Max dependency depth: {max_depth}")


if __name__ == '__main__':
    main()
