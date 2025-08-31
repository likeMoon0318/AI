from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from flask_cors import CORS  # 导入 CORS
import requests
import subprocess
import tempfile
import os
import sys

app = Flask(__name__)
CORS(app)  # 允许所有来源跨域（生产环境建议限定具体域名）
app.secret_key = 'your_secret_key'  # 设置会话密钥

# 模拟用户数据库
users = {
    '202318080314@edu.cn': generate_password_hash('123456'),
    '202318080329@edu.cn': generate_password_hash('123456'),
    '202318080223@edu.cn': generate_password_hash('123456'),
    '202318080318@edu.cn': generate_password_hash('123456'),
    '202318080332@edu.cn': generate_password_hash('123456')
}

def set_video_url(data):
    list = []
    id_list1 = []
    id_list2 = [
        "9DC25B09DA74A0439C33DC5901307461",
        "44DBFCCDF406C8359C33DC5901307461",
        "BA8B63019776CF609C33DC5901307461"
    ]
    url = 'https://www.xuetangx.com/api/v1/lms/service/playurl/{}/?appid=10000'

    for id in id_list2:
        res = requests.get(url.format(id))
        id_list1.append(res.json()['data']['sources']['quality10'][0])

    for i in data:
        list.append(i)

    for i,key in enumerate(list):
        data[key]['video'] = id_list1[i]

        

# 课程数据
courseData = {
    "error-upper": {
        "id": "error-upper",
        "chapter": "第一章 误差分析",
        "title": "1.1 误差（上）",
        "subtitle": "误差的基本概念与分类",
        "icon": "fa-cubes",
        "video": "https://ali-cdn.xuetangx.com/8a3d9abbd5869467-10.mp4?auth_key=1751102634-0-0-be7eca8bb55d5792e417e1b91befa413",
        "keyPoints": [
            { "type": "primary", "title": "重点说明", "content": "绝对误差、相对误差、有效数字的定义与计算" },
            { "type": "warning", "title": "难点标记", "content": "误差传播公式的推导与应用场景" }
        ],
        "code": {
            "filename": "error_calculation.py",
            "language": "python",
            "content": """def calculate_error(approx_value, exact_value):
    # 计算绝对误差和相对误差
    absolute_error = abs(approx_value - exact_value)
    if exact_value != 0:
        relative_error = absolute_error / abs(exact_value)
    else:
        relative_error = 0  # 避免除以零
    
    return {
        'absolute_error': absolute_error,
        'relative_error': relative_error
    }

# 示例：计算 π 的近似值误差
approx_pi = 3.1416
exact_pi = 3.141592653589793
error_results = calculate_error(approx_pi, exact_pi)

print("绝对误差:", error_results['absolute_error'])
print("相对误差:", error_results['relative_error'])""",
            "tip": "尝试计算不同近似值的误差，观察有效数字与误差的关系"
        },
        "practice": {
            "title": "练习题",
            "content": "已知近似值 x=2.718，精确值为 e=2.718281828...，计算：<br>1. 绝对误差<br>2. 相对误差<br>3. 有效数字位数",
            "questions": [
                { "type": "text", "placeholder": "输入你的答案" }
            ]
        },
        "homework": {
            "title": "作业题目",
            "content": "1. 推导一元函数 f(x) = sin(x) 在 x=0 处的误差传播公式<br>2. 假设 x 的近似值误差为 Δx，证明 f(x) 的近似误差约为 |cos(x)|·|Δx|<br>3. 编写代码验证上述结论（取 x=π/4，Δx=0.01）",
            "questions": [
                { "type": "text", "placeholder": "输入推导过程和代码" }
            ]
        }
    },
    "error-lower": {
        "id": "error-lower",
        "chapter": "第一章 误差分析",
        "title": "1.2 误差（下）",
        "subtitle": "误差分析与数值稳定性",
        "icon": "fa-cube",
        "video": "https://ali-cdn.xuetangx.com/ca440b71f6b481f9-10.mp4?auth_key=1751102634-0-0-f466e7de6031bbc996f3d280cf52f849",
        "keyPoints": [
            { "type": "primary", "title": "重点说明", "content": "数值稳定性的定义与判断方法" },
            { "type": "warning", "title": "难点标记", "content": "避免误差放大的计算策略" }
        ],
        "code": {
            "filename": "stability_analysis.py",
            "language": "python",
            "content": """def stable_calculation(n):
    # 稳定的递推公式计算
    result = 0.0
    for i in range(n, 0, -1):
        result = (1.0 - i * result) / i
    return result

def unstable_calculation(n):
    # 不稳定的递推公式计算
    result = 0.0
    for i in range(1, n+1):
        result = (1.0 - i * result)
    return result

# 对比两种方法的稳定性
n = 10
stable = stable_calculation(n)
unstable = unstable_calculation(n)

print(f"稳定方法结果: {stable}")
print(f"不稳定方法结果: {unstable}")""",
            "tip": "尝试增加n的值，观察两种方法的误差变化"
        },
        "practice": {
            "title": "练习题",
            "content": "分析以下计算过程的数值稳定性：<br>计算 I_n = ∫₀¹ xⁿeˣ dx 的递推公式 I_n = e - nIₙ₋₁",
            "questions": [
                { "type": "text", "placeholder": "输入你的分析" }
            ]
        },
        "homework": {
            "title": "作业题目",
            "content": "1. 设计一个稳定的算法计算 ln(100!)<br>2. 比较稳定算法与直接计算的误差差异<br>3. 解释误差产生的原因",
            "questions": [
                { "type": "text", "placeholder": "输入算法设计和分析" }
            ]
        }
    },
    "gauss-elimination-upper": {
        "id": "gauss-elimination-upper",
        "chapter": "第二章 解线性方程组的直接法",
        "title": "2.1 Gauss消去法（上）",
        "subtitle": "基本原理与前向消元",
        "icon": "fa-calculator",
        "video": "https://ali-cdn.xuetangx.com/4703ae3dfa3baa45-10.mp4?auth_key=1751102634-0-0-d704dafdba1b4185bec405b3509e7133",
        "keyPoints": [
            { "type": "primary", "title": "重点说明", "content": "Gauss消去法的基本步骤与矩阵变换" },
            { "type": "warning", "title": "难点标记", "content": "主元选择的重要性" }
        ],
        "code": {
            "filename": "gaussian_elimination.py",
            "language": "python",
            "content": """import numpy as np

def gaussian_elimination(A, b):
    # 高斯消元法求解线性方程组 Ax = b
    n = A.shape[0]
    Ab = np.hstack((A, b.reshape(-1, 1)))  # 增广矩阵
    
    # 前向消元
    for i in range(n):
        # 主元归一化
        pivot = Ab[i, i]
        if pivot == 0:
            raise ValueError("矩阵是奇异的，无法求解")
        Ab[i] /= pivot
        
        # 消元
        for j in range(i+1, n):
            factor = Ab[j, i]
            Ab[j] -= factor * Ab[i]
    
    # 回代求解
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, -1]
        for j in range(i+1, n):
            x[i] -= Ab[i, j] * x[j]
    
    return x

# 测试
A = np.array([[2, 1, -1],
              [1, 3, 2],
              [3, 2, 4]])
b = np.array([8, 11, 20])
x = gaussian_elimination(A, b)
print("方程组的解:", x)""",
            "tip": "可以尝试实现列主元高斯消元法以提高数值稳定性"
        },
        "practice": {
            "title": "练习题",
            "content": "使用高斯消元法求解下列线性方程组：<br>2x + y - z = 8<br>x + 3y + 2z = 11<br>3x + 2y + 4z = 20",
            "questions": [
                { "type": "text", "placeholder": "输入你的答案" }
            ]
        },
        "homework": {
            "title": "作业题目",
            "content": "1. 实现列主元高斯消元法<br>2. 比较基本高斯消元法与列主元法的数值稳定性<br>3. 分析主元选择对结果的影响",
            "questions": [
                { "type": "text", "placeholder": "输入代码和分析" }
            ]
        }
    }
}
set_video_url(courseData)

# 代码执行函数，带超时限制
def run_code(code, timeout=10):
    """安全地执行Python代码并返回输出"""
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    # 获取当前Python解释器路径
    python_interpreter = sys.executable
    
    # 准备执行命令
    cmd = [python_interpreter, temp_file]
    
    try:
        # 使用subprocess执行代码，并设置超时
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}  # 设置编码以避免输出问题
        )
        
        # 检查是否有错误
        if result.returncode != 0:
            # 优先返回标准错误输出，如果没有则返回标准输出
            error_output = result.stderr if result.stderr else result.stdout
            if not error_output:
                error_output = f'执行出错，返回代码: {result.returncode}'
            
            return {
                'success': False,
                'output': error_output
            }
        
        return {
            'success': True,
            'output': result.stdout
        }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': f'代码执行超时（超过{timeout}秒）'
        }
    
    except Exception as e:
        return {
            'success': False,
            'output': f'执行过程中发生错误: {str(e)}'
        }
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

# 代码执行API端点
@app.route('/api/execute', methods=['POST'])
def execute_code():
    """接收并执行Python代码"""
    # 检查用户是否已登录
    if 'email' not in session:
        return jsonify({"status": "error", "message": "请先登录"}), 401
    
    # 获取请求中的代码
    data = request.json
    code = data.get('code')
    
    if not code:
        return jsonify({"status": "error", "message": "缺少代码参数"}), 400
    
    # 执行代码
    result = run_code(code)
    
    return jsonify({
        "status": "success" if result['success'] else "error",
        "output": result['output']
    })

@app.route('/')
def index():
    return render_template('首页面.html')

@app.route('/首页面.html')
def index2():
    return render_template('首页面.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email in users and check_password_hash(users[email], password):
            session['email'] = email
            return redirect(url_for('dashboard'))
        else:
            return jsonify({"status": "error", "message": "用户名或密码错误"}), 401
    return render_template('登录页面.html')

@app.route('/register')
def register():
    return render_template('注册页面.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('1.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

# 新增 API 接口，用于返回课程数据
@app.route('/get_course_data', methods=['GET'])
def get_course_data():
    return jsonify(courseData)

# 处理作业提交，调用豆包API
@app.route('/submit_homework', methods=['POST'])
def submit_homework():
    if 'email' not in session:
        return jsonify({"status": "error", "message": "请先登录"}), 401

    data = request.get_json()  # 获取 JSON 数据
    homework_content = data.get('homework_content', '')  # 提取作业内容
    if not homework_content:
        return jsonify({"status": "error", "message": "缺少作业内容"}), 400

    # 豆包API配置
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    API_KEY = "2fc5fab6-92b9-4b1f-ab0e-20c4eeec6eba"  # 注意：请确认θ是否为合法字符
    MODEL_ID = "doubao-1-5-thinking-pro-250415"

    # 构造请求体（按豆包API格式）
    messages = [
        {"role": "system", "content": "你是专业的计算方法作业批改助手，需要对学生提交的作业进行详细批改：\n1. 检查解题思路是否正确\n2. 指出计算过程中的错误\n3. 提供优化建议\n4. 用中文回复，结果简洁清晰。"},
        {"role": "user", "content": f"作业内容：\n{homework_content}"}
    ]

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.1  # 降低随机性，确保结果稳定
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        # 调用豆包API
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 抛出HTTP错误
        result = response.json()

        # 解析API返回结果
        if "choices" in result and len(result["choices"]) > 0:
            ai_response = result["choices"][0]["message"]["content"]

            # 拆分批改结果和建议（假设AI返回格式为“批改：...\n建议：...”）
            correction = "暂无详细批改"
            suggestion = "暂无优化建议"

            if "批改：" in ai_response and "建议：" in ai_response:
                correction_part, suggestion_part = ai_response.split("建议：", 1)
                correction = correction_part.split("批改：", 1)[-1].strip()
                suggestion = suggestion_part.strip()
            else:
                correction = ai_response
                suggestion = "请补充具体作业内容以获取优化建议"

            return jsonify({
                "status": "success",
                "correction": correction,
                "suggestion": suggestion
            })
        else:
            return jsonify({"status": "error", "message": "API返回结果格式异常"}), 500

    except requests.RequestException as e:
        error_msg = f"API调用失败: {str(e)}"
        if hasattr(e, 'response') and e.response.status_code == 401:
            error_msg = "API认证失败，请检查API密钥"
        return jsonify({"status": "error", "message": error_msg}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"处理作业时出错: {str(e)}"}), 500
    
# 处理课堂小练批改，调用豆包API
@app.route('/api/grade-practice', methods=['POST'])
def grade_practice():
    if 'email' not in session:
        return jsonify({"status": "error", "message": "请先登录"}), 401
    
    data = request.get_json()  # 获取JSON数据
    practice_content = data.get('practice_content', '')  # 提取作业内容
    
    if not practice_content:
        return jsonify({"status": "error", "message": "缺少课堂小练答案"}), 400
    
    # 豆包API配置
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    API_KEY = "2fc5fab6-92b9-4b1f-ab0e-20c4eeec6eba"  # 注意：请确认API密钥有效性
    MODEL_ID = "doubao-1-5-thinking-pro-250415"
    
    # 构造请求体（按豆包API格式）
    messages = [
        {"role": "system", "content": "你是专业的计算方法课堂小练批改助手，需要对学生提交的答案进行详细批改：\n1. 检查答案正确性\n2. 指出解题过程中的错误\n3. 提供简洁明了的优化建议\n4. 用中文回复，结果分点列出。"},
        {"role": "user", "content": f"课堂小练答案：\n{practice_content}"}
    ]
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.1  # 降低随机性，确保结果稳定
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        # 调用豆包API
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 抛出HTTP错误
        result = response.json()
        
        # 解析API返回结果
        if "choices" in result and len(result["choices"]) > 0:
            ai_response = result["choices"][0]["message"]["content"]
            
            # 拆分批改结果和建议（假设AI返回格式为“批改：...\n建议：...”）
            correction = "暂无详细批改"
            suggestion = "暂无优化建议"
            
            if "批改：" in ai_response and "建议：" in ai_response:
                correction_part, suggestion_part = ai_response.split("建议：", 1)
                correction = correction_part.split("批改：", 1)[-1].strip()
                suggestion = suggestion_part.strip()
            else:
                correction = ai_response
                suggestion = "请补充具体答案以获取优化建议"
            
            return jsonify({
                "status": "success",
                "correction": correction,
                "suggestion": suggestion
            })
        else:
            return jsonify({"status": "error", "message": "API返回结果格式异常"}), 500
            
    except requests.RequestException as e:
        error_msg = f"API调用失败: {str(e)}"
        if hasattr(e, 'response') and e.response.status_code == 401:
            error_msg = "API认证失败，请检查API密钥"
        return jsonify({"status": "error", "message": error_msg}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"处理课堂小练时出错: {str(e)}"}), 500
    
# AI助手
@app.route('/api/ai-assistant', methods=['POST'])
def ai_assistant():
    if 'email' not in session:
        return jsonify({"status": "error", "message": "请先登录"}), 401

    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({"status": "error", "message": "缺少问题内容"}), 400

    # 豆包API配置
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    API_KEY = "2fc5fab6-92b9-4b1f-ab0e-20c4eeec6eba"  # 注意：请确认API密钥有效性
    MODEL_ID = "doubao-1-5-thinking-pro-250415"

    # 构造请求体（按豆包API格式）
    messages = [
        {"role": "system", "content": "你是一个通用的AI助手，能回答各种问题。请用中文简洁清晰地回复。"},
        {"role": "user", "content": question}
    ]

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.1  # 降低随机性，确保结果稳定
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        # 调用豆包API
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 抛出HTTP错误
        result = response.json()

        # 解析API返回结果
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"]
            return jsonify({
                "status": "success",
                "answer": answer
            })
        else:
            return jsonify({"status": "error", "message": "API返回结果格式异常"}), 500

    except requests.RequestException as e:
        error_msg = f"API调用失败: {str(e)}"
        if hasattr(e, 'response') and e.response.status_code == 401:
            error_msg = "API认证失败，请检查API密钥"
        return jsonify({"status": "error", "message": error_msg}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"处理问题时出错: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
    #ngrok http http://localhost:5000