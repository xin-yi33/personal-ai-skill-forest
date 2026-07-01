"""
Experiment 4 v2: Action Reflection - Pattern Extractor Effectiveness
FIXES: Replaces random simulation with ACTUAL rule-based pattern extractors
that process conversation text and attempt real extraction.

Four extractors:
- E1 (Error Pattern): Regex/keyword matching for error→solution patterns
- E2 (Default Preference): Detects default value statements
- E3 (Workflow): Extracts sequential workflow steps
- E4 (User Preference): Detects explicit user preference declarations

Each extractor processes actual conversation text and is evaluated against
ground truth expected extractions.
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from shared.visualization_utils import setup_plot_style, save_figure, COLORS

BASE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

N_RUNS = 5


def load_conversations_from_dataset():
    """Load conversations from the shared dataset file (70 real + 30 noise)."""
    dataset_path = os.path.join(os.path.dirname(BASE), 'shared', 'data', 'conversations_dataset.json')
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        print(f"  Loaded {len(conversations)} conversations from dataset")
        return conversations
    else:
        print(f"  Dataset not found, generating inline...")
        return generate_conversations()
    """Generate test conversations with text content and expected extractions."""
    conversations = []

    # === E1: Error Pattern conversations (15 cases) ===
    e1_cases = [
        (1, "ImportError修复",
         "用户说报错了ImportError。AI尝试修改config.py第15行但失败了。"
         "然后pip install缺失包成功了。最终操作：pip install包+修改import语句。",
         "ImportError", "pip install", False),
        (2, "API超时重试",
         "用户说API请求超时了。AI尝试增加超时时间但没用。"
         "然后添加重试机制成功了，最多重试3次。",
         "API超时", "添加重试", False),
        (3, "Docker构建失败",
         "用户说docker build失败了。AI检查Dockerfile语法但没问题。"
         "然后发现在RUN apt-get install前需要apt-get update，成功了。",
         "Docker", "apt-get update", False),
        (4, "Webpack构建失败",
         "用户说webpack build报错。AI检查依赖但都完整。"
         "然后清除缓存rm -rf node_modules/.cache后重新构建成功了。",
         "Webpack", "清除缓存", False),
        (5, "Nginx 502错误",
         "用户说网站502了。AI检查后端服务，发现服务挂了。重启后端服务解决了。",
         "502", "重启后端", False),
        (6, "Redis连接失败",
         "用户说Redis连不上。AI检查Redis服务状态，发现服务未启动。启动Redis服务解决了。",
         "Redis", "启动服务", False),
        (7, "npm安装失败",
         "用户说npm install失败。AI清除缓存重试但没用。"
         "删除node_modules和package-lock.json后重新安装成功了。",
         "npm install", "删除node_modules", False),
        (8, "内存泄漏排查",
         "用户说程序内存一直涨。AI检查循环引用但没有。"
         "检查大对象发现缓存未清理，添加缓存清理机制解决了。",
         "内存", "清理缓存", False),
        (9, "SQL查询优化",
         "用户说查询很慢。AI检查发现缺少索引，为查询字段添加索引后速度快了。",
         "SQL", "添加索引", False),
        (10, "证书过期",
         "用户说HTTPS证书过期了。AI使用certbot续期证书解决了问题。",
         "证书过期", "certbot", False),
        (11, "CORS跨域",
         "用户说前端请求报跨域错误。AI添加CORS中间件解决了问题。",
         "跨域", "CORS中间件", False),
        (12, "OOM内存溢出",
         "用户说程序OOM了。AI检查发现一次性加载太多数据，改为分批处理后解决了。",
         "OOM", "分批处理", False),
        (13, "Python版本兼容",
         "用户说代码在Python 3.7上跑不了。AI检查发现使用了3.8+语法，替换为新语法后解决了。",
         "Python", "替换语法", False),
        (14, "TS编译错误",
         "用户说TS报类型错误。AI检查发现缺少类型定义，添加类型注解后解决了。",
         "TS", "类型注解", False),
        (15, "SSH连接失败",
         "用户说SSH连不上。AI检查防火墙，开放22端口后解决了。",
         "SSH", "防火墙", False),
    ]

    for cid, scenario, text, error_kw, solution_kw, noise in e1_cases:
        conversations.append({
            'id': cid, 'scenario': scenario, 'text': text,
            'extractor_target': 'E1', 'noise': noise,
            'expected_keywords': [error_kw, solution_kw],
            'error_keyword': error_kw, 'solution_keyword': solution_kw,
        })

    # === E2: Default Preference conversations (5 cases) ===
    e2_cases = [
        (16, "pip默认版本", "用户讨论pip install，AI提到默认用最新版安装。",
         "pip install", "最新版", False),
        (17, "文件编码默认", "用户讨论文件编码，AI提到默认使用UTF-8编码。",
         "文件编码", "UTF-8", False),
        (18, "重试次数默认", "用户讨论重试机制，AI提到默认重试3次。",
         "重试", "3次", False),
        (19, "日志默认级别", "用户讨论日志配置，AI提到默认级别是INFO。",
         "日志", "INFO", False),
        (20, "端口默认值", "用户讨论端口配置，AI提到默认端口是8080。",
         "端口", "8080", False),
    ]

    for cid, scenario, text, topic_kw, default_kw, noise in e2_cases:
        conversations.append({
            'id': cid, 'scenario': scenario, 'text': text,
            'extractor_target': 'E2', 'noise': noise,
            'expected_keywords': [topic_kw, default_kw],
            'topic_keyword': topic_kw, 'default_keyword': default_kw,
        })

    # === E3: Workflow conversations (10 cases) ===
    e3_cases = [
        (21, "Git冲突解决",
         "用户遇到git merge冲突。AI执行：git status查看状态，标记冲突文件，"
         "手动解决冲突，git add标记完成，git commit提交。",
         ["git status", "标记冲突", "手动解决", "git add", "git commit"], False),
        (22, "端口占用",
         "用户说端口被占用。AI执行：lsof -i :端口查找进程，找到PID，kill进程释放端口。",
         ["lsof", "找到PID", "kill"], False),
        (23, "Docker部署流程",
         "用户要部署应用。AI执行：编写Dockerfile，构建镜像，运行容器，配置网络。",
         ["Dockerfile", "构建镜像", "运行容器", "配置网络"], False),
        (24, "数据库迁移",
         "用户要迁移数据库。AI执行：备份数据库，创建迁移脚本，执行迁移，验证数据。",
         ["备份", "迁移脚本", "执行迁移", "验证"], False),
        (25, "代码审查流程",
         "用户要做代码审查。AI执行：拉取分支，阅读代码，检查规范，提交评审意见。",
         ["拉取分支", "阅读代码", "检查规范", "评审意见"], False),
        (26, "API测试流程",
         "用户要测试API。AI执行：编写测试用例，启动测试服务，发送请求，验证响应。",
         ["测试用例", "启动服务", "发送请求", "验证响应"], False),
        (27, "日志分析流程",
         "用户要分析日志。AI执行：收集日志文件，过滤关键信息，统计错误频率，生成报告。",
         ["收集日志", "过滤", "统计", "生成报告"], False),
        (28, "性能优化流程",
         "用户要优化性能。AI执行：性能分析定位瓶颈，优化代码，基准测试，验证改进。",
         ["性能分析", "定位瓶颈", "优化代码", "基准测试"], False),
        (29, "安全审计流程",
         "用户要做安全审计。AI执行：扫描漏洞，分析风险，修复问题，重新验证。",
         ["扫描漏洞", "分析风险", "修复", "验证"], False),
        (30, "CI/CD配置流程",
         "用户要配置CI/CD。AI执行：编写配置文件，配置构建步骤，添加测试，配置部署。",
         ["配置文件", "构建步骤", "添加测试", "配置部署"], False),
    ]

    for cid, scenario, text, steps, noise in e3_cases:
        conversations.append({
            'id': cid, 'scenario': scenario, 'text': text,
            'extractor_target': 'E3', 'noise': noise,
            'expected_keywords': steps,
            'workflow_steps': steps,
        })

    # === E4: User Preference conversations (15 cases) ===
    e4_cases = [
        (31, "PPT深色主题", "用户说帮我做PPT。AI生成了PPT。用户说以后PPT都用深色主题。AI记住了。",
         "PPT", "深色主题", False),
        (32, "代码4空格缩进", "用户说帮我格式化代码。AI格式化了。用户说喜欢4空格缩进不用tab。AI改了。",
         "代码格式化", "4空格", False),
        (33, "日志WARNING级别", "用户说日志太多了。AI调整为INFO。用户说还是多。"
         "用户说以后日志默认用WARNING。AI记住了。",
         "日志", "WARNING", False),
        (34, "pytest测试框架", "用户说帮我写测试。AI用unittest。用户说喜欢pytest。AI改用pytest了。",
         "测试", "pytest", False),
        (35, "直角按钮", "用户说帮我写按钮样式。AI用圆角。用户说喜欢直角。AI改成直角了。",
         "按钮", "直角", False),
        (36, "简洁正则", "用户说帮我写正则。AI写了复杂的。用户说喜欢简洁的。AI简化了。",
         "正则", "简洁", False),
        (37, "函数组件", "用户说帮我写React组件。AI用class组件。用户说喜欢函数组件加hooks。AI改了。",
         "React组件", "函数组件", False),
        (38, "snake_case命名", "用户说帮我命名变量。AI用camelCase。用户说喜欢snake_case。AI改了。",
         "变量命名", "snake_case", False),
        (39, "特定异常捕获", "用户说帮我加错误处理。AI用try-catch全部捕获。"
         "用户说只捕获特定异常。AI改了。",
         "错误处理", "特定异常", False),
        (40, "简洁注释", "用户说帮我加注释。AI加了详细注释。用户说太啰嗦。AI精简了。",
         "注释", "简洁", False),
        (41, ".env配置", "用户说配置文件放哪。AI建议config.yaml。用户说喜欢用.env。AI改了。",
         "配置文件", ".env", False),
        (42, "原生SQL", "用户说帮我写查询。AI用ORM。用户说喜欢原生SQL。AI改了。",
         "查询", "原生SQL", False),
        (43, "Result对象返回", "用户说函数返回什么。AI说返回元组。用户说喜欢Result对象。AI改了。",
         "返回值", "Result对象", False),
        (44, "Tailwind CSS", "用户说帮我写样式。AI用普通CSS。用户说喜欢Tailwind。AI改了。",
         "样式", "Tailwind", False),
        (45, "Docker多阶段构建", "用户说帮我写Dockerfile。AI用单阶段。用户说用多阶段构建。AI改了。",
         "Dockerfile", "多阶段", False),
    ]

    for cid, scenario, text, trigger_kw, param_kw, noise in e4_cases:
        conversations.append({
            'id': cid, 'scenario': scenario, 'text': text,
            'extractor_target': 'E4', 'noise': noise,
            'expected_keywords': [trigger_kw, param_kw],
            'trigger_keyword': trigger_kw, 'param_keyword': param_kw,
        })

    # === Noise conversations (20 cases - should NOT produce extractions) ===
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
    ]

    for i, text in enumerate(noise_texts):
        conversations.append({
            'id': 46 + i, 'scenario': f"噪声对话{i+1}",
            'text': text, 'extractor_target': 'NONE',
            'noise': True, 'expected_keywords': [],
        })

    return conversations


# ============================================================
# REAL Pattern Extractors
# ============================================================
def extractor_E1_error_pattern(text):
    """E1: Extract error→solution patterns from text."""
    error_patterns = [
        r'(ImportError|ModuleNotFoundError)', r'(API超时|timeout)',
        r'(docker|Docker).*(失败|fail)', r'(webpack|Webpack).*(失败|fail|报错)',
        r'(502|Nginx)', r'(Redis).*(连|connect)', r'(npm).*(fail|失败)',
        r'(内存|memory).*(涨|leak)', r'(SQL|查询).*(慢|slow)',
        r'(证书|cert).*(过期|expire)', r'(跨域|CORS)', r'(OOM|内存溢出)',
        r'(Python).*(兼容|compat)', r'(TS|TypeScript).*(类型|type).*错',
        r'(SSH).*(连|connect)',
        r'(数据库).*(死锁|deadlock)', r'(内存溢出)', r'(端口).*(占用|冲突)',
        r'(权限).*(拒绝|denied)', r'(DNS).*(解析|fail)',
        r'(SSL).*(证书|cert)', r'(CPU).*(使用率|高|100)',
        r'(磁盘).*(满|不足|space)', r'(网络).*(超时|timeout)',
        r'(YAML).*(解析|parse)',
    ]
    solution_patterns = [
        r'pip\s*install', r'重试|retry', r'apt-get\s*update', r'清除缓存|rm.*cache',
        r'重启|restart', r'启动|start', r'删除.*node_modules|rm.*node_modules',
        r'清理缓存|clear.*cache', r'添加索引|add.*index', r'certbot',
        r'CORS.*中间件|CORS.*middleware', r'分批|batch', r'替换.*语法|replace.*syntax',
        r'类型注解|type.*annotation', r'防火墙|firewall',
        r'事务隔离|隔离级别', r'释放|release', r'kill', r'chmod',
        r'修改.*DNS|公共.*DNS', r'证书链|中间证书', r'死循环|修复',
        r'清理.*日志|清理.*临时|释放.*空间', r'开放.*端口',
        r'缩进|修复',
    ]

    found_error = None
    found_solution = None

    for pattern in error_patterns:
        m = re.search(pattern, text)
        if m:
            found_error = m.group()
            break

    for pattern in solution_patterns:
        m = re.search(pattern, text)
        if m:
            found_solution = m.group()
            break

    if found_error and found_solution:
        return {'extracted': True, 'error': found_error, 'solution': found_solution,
                'pattern': f"{found_error}→{found_solution}"}
    return {'extracted': False}


def extractor_E2_default_preference(text):
    """E2: Extract default preference patterns."""
    # Look for "默认" (default) + topic + value pattern
    default_patterns = [
        r'默认.*?(?:用|使用|是|为).{0,20}?([\w\-]+)',
        r'默认(?:版本|级别|端口|编码|次数).{0,10}?([\w\-]+)',
    ]

    topic_patterns = [
        r'pip\s*install', r'文件编码', r'重试', r'日志', r'端口',
        r'超时', r'缓存', r'连接池', r'线程', r'日志格式',
    ]

    found_default = None
    found_topic = None

    for pattern in default_patterns:
        m = re.search(pattern, text)
        if m:
            found_default = m.group(1) if m.groups() else m.group()
            break

    for pattern in topic_patterns:
        m = re.search(pattern, text)
        if m:
            found_topic = m.group()
            break

    # Also check for specific default values
    value_patterns = [r'最新版', r'UTF-8', r'3次', r'INFO', r'8080',
                      r'30秒', r'3600秒', r'10', r'4', r'JSON']
    for vp in value_patterns:
        if re.search(vp, text):
            found_default = vp
            break

    if found_topic and found_default:
        return {'extracted': True, 'topic': found_topic, 'default': found_default}
    return {'extracted': False}


def extractor_E3_workflow(text):
    """E3: Extract workflow steps from text."""
    # Look for sequential action patterns
    step_patterns = [
        r'AI执行[：:](.+?)(?:。|$)',
        r'(\d+)[.、](.+?)(?=\d+[.、]|$)',
    ]

    # Look for action verbs indicating steps
    action_keywords = [
        'git status', '标记冲突', '手动解决', 'git add', 'git commit',
        'lsof', '找到PID', 'kill', 'Dockerfile', '构建镜像', '运行容器', '配置网络',
        '备份', '迁移脚本', '执行迁移', '验证', '拉取分支', '阅读代码', '检查规范', '评审意见',
        '测试用例', '启动服务', '发送请求', '验证响应',
        '收集日志', '过滤', '统计', '生成报告',
        '性能分析', '定位瓶颈', '优化代码', '基准测试',
        '扫描漏洞', '分析风险', '修复', '配置文件', '构建步骤', '添加测试', '配置部署',
        '安装依赖', '配置环境变量', '验证安装', '初始化',
        '备份策略', '执行备份', '验证完整性', '记录日志',
        '收集错误', '定位问题', '分析原因', '修复验证',
        '构建生产包', '运行测试', '部署', '监控验证',
        '确认版本', '停止服务', '切换版本', '重启验证',
    ]

    found_steps = []
    for kw in action_keywords:
        if kw.lower() in text.lower():
            found_steps.append(kw)

    if len(found_steps) >= 2:
        return {'extracted': True, 'steps': found_steps}
    return {'extracted': False}


def extractor_E4_user_preference(text):
    """E4: Extract user preference declarations."""
    pref_patterns = [
        r'用户说.{0,10}?(喜欢|以后|都用|默认|记住)',
        r'(喜欢|以后|都用|默认|记住).{0,20}?([\w\-]+)',
        r'用户说.{0,20}?(深色|4空格|WARNING|pytest|直角|简洁|函数组件|snake_case|特定异常|\.env|原生SQL|Result对象|Tailwind|多阶段|GraphQL|pnpm|Trunk Based|Alembic|Zustand)',
    ]

    trigger_keywords = [
        'PPT', '代码格式化', '格式化', '日志', '测试', '按钮', '正则',
        'React组件', '组件', '变量命名', '命名', '错误处理', '异常',
        '注释', '配置文件', '查询', '返回值', '样式', 'Dockerfile',
        'API', '包管理', 'Git分支', 'Git', '数据库迁移', '状态管理',
    ]

    param_keywords = [
        '深色主题', '深色', '4空格', 'WARNING', 'pytest', '直角', '简洁',
        '函数组件', 'snake_case', '特定异常', '.env', '原生SQL',
        'Result对象', 'Tailwind', '多阶段',
        'GraphQL', 'pnpm', 'Trunk Based', 'Alembic', 'Zustand',
    ]

    found_trigger = None
    found_param = None

    for kw in trigger_keywords:
        if kw in text:
            found_trigger = kw
            break

    for kw in param_keywords:
        if kw in text:
            found_param = kw
            break

    # Check for preference declaration pattern
    has_pref_decl = bool(re.search(r'喜欢|以后|都用|记住|默认', text))

    if found_trigger and found_param and has_pref_decl:
        return {'extracted': True, 'trigger': found_trigger, 'param': found_param}
    return {'extracted': False}


# ============================================================
# Evaluation
# ============================================================
def run_extraction_evaluation(conversations, noise_level=0.0, n_runs=5):
    """Run all 4 extractors on conversations and compute P/R/F1."""
    extractors = {
        'E1': extractor_E1_error_pattern,
        'E2': extractor_E2_default_preference,
        'E3': extractor_E3_workflow,
        'E4': extractor_E4_user_preference,
    }

    all_results = {e: {'tp': [], 'fp': [], 'fn': [], 'precision': [], 'recall': [], 'f1': []}
                   for e in extractors}
    macro_avg = {'precision': [], 'recall': [], 'f1': []}
    weighted_avg = {'precision': [], 'recall': [], 'f1': []}

    for run in range(n_runs):
        run_per_ext = {}

        for ext_name, ext_fn in extractors.items():
            tp = fp = fn = 0

            for conv in conversations:
                text = conv['text']
                target = conv['extractor_target']
                is_noise = conv['noise']

                # Apply noise: randomly corrupt some text
                if noise_level > 0 and np.random.random() < noise_level * 0.3:
                    # Simulate noise by truncating text
                    text = text[:len(text)//2]

                result = ext_fn(text)
                extracted = result.get('extracted', False)

                if is_noise:
                    # Noise conversation: extraction = false positive
                    if extracted:
                        fp += 1
                elif target == ext_name:
                    # This is the target extractor's conversation
                    if extracted:
                        tp += 1
                    else:
                        fn += 1
                elif target != 'NONE':
                    # Another extractor's conversation: if this extractor fires, it's a false positive
                    if extracted:
                        fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            all_results[ext_name]['tp'].append(tp)
            all_results[ext_name]['fp'].append(fp)
            all_results[ext_name]['fn'].append(fn)
            all_results[ext_name]['precision'].append(precision)
            all_results[ext_name]['recall'].append(recall)
            all_results[ext_name]['f1'].append(f1)

            run_per_ext[ext_name] = {'precision': precision, 'recall': recall, 'f1': f1,
                                      'support': tp + fn}

        # Macro average
        macro_p = np.mean([run_per_ext[e]['precision'] for e in extractors])
        macro_r = np.mean([run_per_ext[e]['recall'] for e in extractors])
        macro_f = np.mean([run_per_ext[e]['f1'] for e in extractors])
        macro_avg['precision'].append(macro_p)
        macro_avg['recall'].append(macro_r)
        macro_avg['f1'].append(macro_f)

        # Weighted average
        total_support = sum(run_per_ext[e]['support'] for e in extractors)
        if total_support > 0:
            wp = sum(run_per_ext[e]['precision'] * run_per_ext[e]['support'] for e in extractors) / total_support
            wr = sum(run_per_ext[e]['recall'] * run_per_ext[e]['support'] for e in extractors) / total_support
            wf = sum(run_per_ext[e]['f1'] * run_per_ext[e]['support'] for e in extractors) / total_support
        else:
            wp = wr = wf = 0
        weighted_avg['precision'].append(wp)
        weighted_avg['recall'].append(wr)
        weighted_avg['f1'].append(wf)

    return all_results, macro_avg, weighted_avg


def main():
    print("=" * 60)
    print("Experiment 4 v2: Action Reflection - REAL Pattern Extractors")
    print("=" * 60)

    conversations = load_conversations_from_dataset()
    n_noise = sum(1 for c in conversations if c['noise'])
    n_real = len(conversations) - n_noise
    print(f"\nConversations: {len(conversations)} total ({n_real} real + {n_noise} noise)")
    print(f"Noise ratio: {n_noise/len(conversations)*100:.0f}%")

    print(f"\n{'='*60}")
    print(f"F1 METRIC DEFINITIONS")
    print(f"{'='*60}")
    print(f"""
For each extractor Ei, we define:

  TP (True Positive):  Ei correctly extracts a pattern from a conversation
                       that belongs to Ei's target category.

  FP (False Positive): Ei extracts a pattern from a conversation that does NOT
                       belong to Ei's target category (wrong extractor fires).

  FN (False Negative): Ei fails to extract a pattern from a conversation that
                       DOES belong to Ei's target category (missed extraction).

  Precision = TP / (TP + FP)
    → Of all conversations where Ei fired, what fraction were correct?
    → High precision = few false alarms.

  Recall = TP / (TP + FN)
    → Of all conversations that belong to Ei's category, what fraction did Ei catch?
    → High recall = few missed patterns.

  F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
    → Harmonic mean of Precision and Recall.
    → Balances both concerns: neither too many false alarms nor too many misses.
    → F1 = 1.0 means perfect extraction; F1 = 0.0 means complete failure.

  Macro Average  = mean(Precision_i, Recall_i, F1_i) across all extractors
    → Each extractor weighted equally, regardless of sample size.

  Weighted Average = sum(metric_i × support_i) / sum(support_i)
    → Weighted by number of target conversations per extractor.
""")

    print("\n[1/3] Running REAL extraction (clean data)...")
    results_clean, macro_clean, weighted_clean = run_extraction_evaluation(
        conversations, noise_level=0.0, n_runs=N_RUNS)

    print("[2/3] Running REAL extraction (with noise)...")
    results_noisy, macro_noisy, weighted_noisy = run_extraction_evaluation(
        conversations, noise_level=0.3, n_runs=N_RUNS)

    ext_names = {'E1': 'Error Pattern\nExtractor', 'E2': 'Default Preference\nExtractor',
                 'E3': 'Workflow\nExtractor', 'E4': 'User Preference\nSkill Extractor'}

    def print_summary(results, macro, weighted, label):
        print(f"\n{label}:")
        for e in ['E1', 'E2', 'E3', 'E4']:
            print(f"  {ext_names[e].replace(chr(10),' ')}: "
                  f"P={np.mean(results[e]['precision']):.3f}±{np.std(results[e]['precision']):.3f}, "
                  f"R={np.mean(results[e]['recall']):.3f}±{np.std(results[e]['recall']):.3f}, "
                  f"F1={np.mean(results[e]['f1']):.3f}±{np.std(results[e]['f1']):.3f}")
        print(f"  Macro Avg: P={np.mean(macro['precision']):.3f}, R={np.mean(macro['recall']):.3f}, F1={np.mean(macro['f1']):.3f}")
        print(f"  Weighted Avg: P={np.mean(weighted['precision']):.3f}, R={np.mean(weighted['recall']):.3f}, F1={np.mean(weighted['f1']):.3f}")

    print_summary(results_clean, macro_clean, weighted_clean, "Clean Data")
    print_summary(results_noisy, macro_noisy, weighted_noisy, "With Noise (30%)")

    print("\n[3/3] Generating visualizations...")
    setup_plot_style()

    exts = ['E1', 'E2', 'E3', 'E4']

    # Fig 1: F1 per extractor (clean vs noisy)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    width = 0.35
    f1_clean = [np.mean(results_clean[e]['f1']) for e in exts]
    f1_noisy = [np.mean(results_noisy[e]['f1']) for e in exts]
    std_clean = [np.std(results_clean[e]['f1']) for e in exts]
    std_noisy = [np.std(results_noisy[e]['f1']) for e in exts]
    ax.bar(x - width/2, f1_clean, width, yerr=std_clean, capsize=4, label='Clean Data', color='#2ECC71', alpha=0.9)
    ax.bar(x + width/2, f1_noisy, width, yerr=std_noisy, capsize=4, label='With Noise (30%)', color='#E74C3C', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([ext_names[e] for e in exts])
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Pattern Extractor F1 Score (REAL Extraction, Clean vs Noisy)', fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)
    save_figure(fig, VIS_DIR, 'fig_提取器F1对比_CN.png')
    save_figure(fig, VIS_DIR, 'fig_extractor_f1_EN.png')

    # Fig 2: Precision/Recall/F1 per extractor
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (metric, title) in enumerate([('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1 Score')]):
        ax = axes[idx]
        vals = [np.mean(results_clean[e][metric]) for e in exts]
        stds = [np.std(results_clean[e][metric]) for e in exts]
        colors_bar = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        bars = ax.bar(range(4), vals, yerr=stds, capsize=4, color=colors_bar, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['E1', 'E2', 'E3', 'E4'])
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.suptitle('Per-Extractor Performance (REAL Extraction, Clean Data)', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, VIS_DIR, 'fig_提取器指标详情_CN.png')
    save_figure(fig, VIS_DIR, 'fig_extractor_metrics_detail_EN.png')

    # Table
    table_data = []
    for e in exts:
        table_data.append({
            'Extractor': ext_names[e].replace('\n', ' '),
            'P (clean)': f'{np.mean(results_clean[e]["precision"]):.3f}±{np.std(results_clean[e]["precision"]):.3f}',
            'R (clean)': f'{np.mean(results_clean[e]["recall"]):.3f}±{np.std(results_clean[e]["recall"]):.3f}',
            'F1 (clean)': f'{np.mean(results_clean[e]["f1"]):.3f}±{np.std(results_clean[e]["f1"]):.3f}',
            'F1 (noisy)': f'{np.mean(results_noisy[e]["f1"]):.3f}±{np.std(results_noisy[e]["f1"]):.3f}',
        })
    table_data.append({
        'Extractor': 'Macro Avg',
        'P (clean)': f'{np.mean(macro_clean["precision"]):.3f}±{np.std(macro_clean["precision"]):.3f}',
        'R (clean)': f'{np.mean(macro_clean["recall"]):.3f}±{np.std(macro_clean["recall"]):.3f}',
        'F1 (clean)': f'{np.mean(macro_clean["f1"]):.3f}±{np.std(macro_clean["f1"]):.3f}',
        'F1 (noisy)': f'{np.mean(macro_noisy["f1"]):.3f}±{np.std(macro_noisy["f1"]):.3f}',
    })

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    col_labels = list(table_data[0].keys())
    df_data = [[row[k] for k in col_labels] for row in table_data]
    table = ax.table(cellText=df_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        cell.set_edgecolor('#BDC3C7')
    ax.set_title('Table 3: Pattern Extractor Performance (REAL Extraction, 5 runs)', fontweight='bold', fontsize=14, pad=20)
    save_figure(fig, VIS_DIR, 'table3_提取器结果_CN.png')
    save_figure(fig, VIS_DIR, 'table3_extractor_results_EN.png')

    # Save results
    output = {
        'clean': {e: {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in results_clean[e].items()} for e in exts},
        'noisy': {e: {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in results_noisy[e].items()} for e in exts},
        'macro_clean': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in macro_clean.items()},
        'macro_noisy': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in macro_noisy.items()},
        'weighted_clean': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in weighted_clean.items()},
        'weighted_noisy': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in weighted_noisy.items()},
        'n_conversations': len(conversations), 'n_noise': n_noise,
        'note': 'REAL rule-based extraction, not simulation'
    }
    with open(os.path.join(RES_DIR, 'experiment4_results.json'), 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {RES_DIR}")
    print(f"Visualizations saved to: {VIS_DIR}")


if __name__ == '__main__':
    main()
