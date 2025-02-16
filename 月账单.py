import datetime
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from wordcloud import WordCloud
import jieba
from pathlib import Path
import re
import platform
import logging
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.font_manager as fm

class ExpenseTracker:
    """个人记账系统，支持支出记录、分类统计和可视化分析。

    特点：
    - 自动分类支出
    - 数据持久化存储
    - 月度和年度报告
    - 支出趋势分析
    - 消费词云展示
    """

    def __init__(self, data_file: str = 'expenses.json'):
        """初始化记账系统。

        Args:
            data_file: JSON 数据文件路径
        """
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 定义分类关键词
        self.category_keywords = {
            '吃饭': ['饭', '餐', '食', '面', '火锅', '烧烤', '小吃', '菜', '米', '奶茶',
                     '饮料', '水果', '早餐', '午餐', '晚餐', '夜宵', '零食', '外卖',
                     '堂食', '快餐', '麦当劳', '肯德基'],
            '娱乐': ['电影', '游戏', 'ktv', '唱歌', '电玩', '网吧', '电竞', '演唱会',
                     '音乐会', '剧场', '展览', '博物馆', '健身', '运动', '旅游', '景点',
                     '门票', '电视会员', '视频会员'],
            '交通': ['地铁', '公交', '出租', '打车', '单车', '共享单车', '加油', '停车',
                     '高铁', '火车', '飞机', '机票', '车票'],
            '购物': ['衣服', '裤子', '鞋', '包', '化妆品', '护肤品', '电子产品', '手机',
                     '电脑', '数码', '家具', '日用品', '超市'],
            '其他': []  # 默认分类
        }

        # 这里将 self.expenses 初始化为嵌套的 defaultdict
        self.expenses = defaultdict(lambda: defaultdict(list))
        self.filename = data_file
        self.load_data()

    def load_data(self) -> None:
        """从 JSON 文件加载支出数据。"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    # 已经初始化为 defaultdict，这里继续使用
                    loaded_data = json.load(f)
                    for month, categories in loaded_data.items():
                        for category, expense_list in categories.items():
                            self.expenses[month][category].extend(expense_list)
                self.logger.info("数据加载成功")
        except Exception as e:
            self.logger.error(f"加载数据时出错: {str(e)}")
            # 出错时也确保 expenses 为 defaultdict
            self.expenses = defaultdict(lambda: defaultdict(list))

    def save_data(self) -> None:
        """将支出数据保存到 JSON 文件。"""
        try:
            # 转换为普通字典后保存
            regular_dict = {
                month: dict(categories)
                for month, categories in self.expenses.items()
            }
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(regular_dict, f, ensure_ascii=False, indent=2)
            self.logger.info("数据保存成功")
        except Exception as e:
            self.logger.error(f"保存数据时出错: {str(e)}")

    def categorize_description(self, description: str) -> str:
        """根据描述自动分类支出。

        Args:
            description: 支出描述

        Returns:
            str: 支出类别
        """
        description = description.lower()
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in description:
                    return category
        return '其他'

    def add_expense(self, date_str: str, description: str,
                    amount: float, category: Optional[str] = None) -> bool:
        """添加一条支出记录。

        Args:
            date_str: 日期字符串，格式为YYYY/MM/DD
            description: 支出描述
            amount: 支出金额
            category: 支出类别（可选）

        Returns:
            bool: 是否添加成功
        """
        try:
            # 如果日期年份部分为两位，则补全为四位年份
            if len(date_str.split('/')[0]) == 2:
                date_str = '20' + date_str
            date = datetime.datetime.strptime(date_str, '%Y/%m/%d')
            month_key = f"{date.year}-{date.month:02d}"

            if category is None:
                category = self.categorize_description(description)

            expense = {
                'date': date_str,
                'description': description,
                'amount': amount,
                'category': category
            }

            # 这里不会再出现 KeyError，因为 self.expenses 已是 defaultdict
            self.expenses[month_key][category].append(expense)
            self.save_data()
            self.logger.info(f"记账成功: {description} 已归类到 {category}")
            return True
        except ValueError as e:
            self.logger.error(f"日期格式错误: {str(e)}")
            print("请使用YYYY/MM/DD 格式的日期")
            return False
        except Exception as e:
            self.logger.error(f"添加支出时出错: {str(e)}")
            print(f"添加支出时出错: {str(e)}")
            return False

    def parse_input(self, input_str: str) -> Optional[Tuple[str, str, float, None]]:
        """解析用户输入的支出信息。

        Args:
            input_str: 用户输入字符串，格式为 "YYYY/MM/DD，描述-金额"

        Returns:
            Tuple[str, str, float, None] or None: 解析结果
        """
        # 使用正则表达式解析输入
        pattern = r'(\d{2,4}/\d{1,2}/\d{1,2})，(.+)-(\d+\.?\d*)'
        match = re.match(pattern, input_str)

        if not match:
            print("输入格式错误，请使用'YYYY/MM/DD，描述-金额'的格式")
            print("例如：2024/02/14，黄焖鸡米饭-25")
            return None

        try:
            date_str, description, amount = match.groups()
            amount = float(amount)
            return date_str, description, amount, None
        except ValueError:
            print("金额格式错误，请输入有效的数字")
            return None
        except Exception as e:
            self.logger.error(f"解析输入时出错: {str(e)}")
            print(f"解析输入时出错: {str(e)}")
            return None

    def get_monthly_summary(self, month_key: str) -> Tuple[Dict[str, float], float]:
        """获取月度支出摘要。

        Args:
            month_key: 月份键值，格式为YYYY-MM

        Returns:
            Tuple[Dict[str, float], float]: (各类别总额字典, 总支出)
        """
        monthly_data = self.expenses.get(month_key, {})
        total = 0.0
        category_totals = {}

        for category in self.category_keywords.keys():
            category_total = sum(
                expense['amount']
                for expense in monthly_data.get(category, [])
            )
            category_totals[category] = category_total
            total += category_total

        return category_totals, total

    def get_font_path(self) -> str:
        """根据操作系统获取合适的字体路径。

        Returns:
            str: 字体文件路径
        """
        system = platform.system()
        if system == 'Windows':
            return "C:\Windows\Fonts\simhei.ttf"
        elif system == 'Darwin':  # macOS
            return "/System/Library/Fonts/PingFang.ttc"
        else:  # Linux
            # 可以添加更多的字体路径检查
            possible_paths = [
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            raise FileNotFoundError("未找到合适的中文字体文件")

    def generate_word_cloud(self, month_key: str) -> None:
        """生成并显示月度消费词云。

        Args:
            month_key: 月份键值，格式为YYYY-MM
        """
        try:
            font_path = self.get_font_path()
            font_prop = fm.FontProperties(fname=font_path)

            # 收集该月所有消费描述
            text = ""
            monthly_data = self.expenses.get(month_key, {})
            for category in monthly_data.values():
                for expense in category:
                    text += expense['description'] + " "

            if not text.strip():
                print(f"{month_key} 月没有消费记录")
                return

            # 使用结巴分词
            words = jieba.cut(text)
            text = " ".join(words)


            # 创建词云
            wordcloud = WordCloud(
                font_path=font_path,
                width=800,
                height=400,
                background_color='white',
                max_words=100
            ).generate(text)

            # 显示词云
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"{month_key} 月消费关键词", fontproperties=font_prop) # 应用字体
            plt.show()
        except FileNotFoundError:
            self.logger.error("未找到合适的中文字体文件")
            print("无法生成词云：未找到合适的中文字体文件")
            return
        except Exception as e:
            self.logger.error(f"生成词云时出错: {str(e)}")
            print(f"生成词云时出错: {str(e)}")

    def generate_monthly_report(self, month_key: str) -> None:
        """生成月度报告，包括饼图、词云和环比分析。

        Args:
            month_key: 月份键值，格式为YYYY-MM
        """
        try:
            font_path = self.get_font_path()
            font_prop = fm.FontProperties(fname=font_path)
            category_totals, total = self.get_monthly_summary(month_key)

            if total == 0:
                print(f"{month_key} 月没有记录")
                return

            # 生成饼图
            plt.figure(figsize=(10, 6))
            categories = [cat for cat, amount in category_totals.items() if amount > 0]
            amounts = [amount for amount in category_totals.values() if amount > 0]

            plt.pie(amounts,
                    labels=[f"{cat}\n{amount:.2f}元" for cat, amount in zip(categories, amounts)],
                    autopct='%1.1f%%',
                    textprops={'fontproperties': font_prop}) # 饼图标签应用字体
            plt.title(f"{month_key} 月消费占比分析", fontproperties=font_prop) # 应用字体
            plt.show()

            # 生成词云
            self.generate_word_cloud(month_key)

            # 计算环比
            year, month = map(int, month_key.split('-'))
            prev_month = f"{year}-{month - 1:02d}" if month > 1 else f"{year - 1}-12"
            prev_totals, _ = self.get_monthly_summary(prev_month)

            print(f"\n{month_key} 月消费总额: {total:.2f}元")
            print("\n环比变化:")
            for category in self.category_keywords.keys():
                current = category_totals.get(category, 0)
                previous = prev_totals.get(category, 0)
                if previous == 0:
                    change = "无上月数据"
                else:
                    change = f"{((current - previous) / previous * 100):.1f}%"
                if current > 0 or previous > 0:  # 只显示有变化的类别
                    print(f"{category}: {change}")
        except FileNotFoundError:
            self.logger.error("未找到合适的中文字体文件")
            print("生成月度报告失败：未找到合适的中文字体文件。")
            return
        except Exception as e:
            self.logger.error(f"生成月度报告时出错: {str(e)}")
            print(f"生成月度报告时出错: {str(e)}")

    def generate_yearly_report(self, year: str) -> None:
        """生成年度报告，包括趋势图和消费分析。

        Args:
            year: 年份
        """
        try:
            font_path = self.get_font_path()
            font_prop = fm.FontProperties(fname=font_path)
            yearly_data = defaultdict(lambda: defaultdict(float))
            monthly_totals = []

            for month in range(1, 13):
                month_key = f"{year}-{month:02d}"
                category_totals, total = self.get_monthly_summary(month_key)

                for category in self.category_keywords.keys():
                    yearly_data[category][month] = category_totals.get(category, 0)
                monthly_totals.append(total)

            # 检查是否有数据
            if sum(monthly_totals) == 0:
                print(f"{year}年没有记录")
                return

            # 生成月度趋势图
            plt.figure(figsize=(12, 6))
            months = range(1, 13)
            for category in self.category_keywords.keys():
                values = [yearly_data[category][m] for m in months]
                if sum(values) > 0:  # 只显示有数据的类别
                    plt.plot(months, values, label=category, marker='o')

            plt.title(f"{year}年消费趋势分析", fontproperties=font_prop) # 应用字体
            plt.xlabel("月份", fontproperties=font_prop) # 应用字体
            plt.ylabel("金额（元）", fontproperties=font_prop) # 应用字体
            plt.legend(prop=font_prop) # 图例应用字体
            plt.grid(True)
            plt.show()

            # 计算年度总结
            total_year = sum(monthly_totals)

            print(f"\n{year}年消费情感分析报告:")
            print(f"年度总支出: {total_year:.2f}元")

            # 消费类别分析
            category_year_totals = {
                cat: sum(yearly_data[cat].values())
                for cat in self.category_keywords.keys()
            }
            max_category = max(
                category_year_totals.items(),
                key=lambda x: x[1]
            )
            print(f"\n主要支出类别: {max_category[0]}，"
                  f"占比 {(max_category[1] / total_year * 100):.1f}%")

            # 月均消费分析
            monthly_avg = total_year / 12
            high_months = [m for m in range(1, 13)
                           if monthly_totals[m - 1] > monthly_avg]
            print(f"月平均支出: {monthly_avg:.2f}元")
            if high_months:
                print(f"消费较高的月份: {', '.join(str(m) for m in high_months)}月")

            # 消费趋势分析
            trend = ((monthly_totals[-1] - monthly_totals[0]) / monthly_totals[0]
                     if monthly_totals[0] != 0 else 0)
            if trend > 0.1:
                print("消费趋势: 呈上升趋势，建议关注支出控制")
            elif trend < -0.1:
                print("消费趋势: 呈下降趋势，支出控制良好")
            else:
                print("消费趋势: 基本稳定")

        except FileNotFoundError:
            self.logger.error("未找到合适的中文字体文件")
            print("生成年度报告失败：未找到合适的中文字体文件。")
            return

        except Exception as e:
            self.logger.error(f"生成年度报告时出错: {str(e)}")
            print(f"生成年度报告时出错: {str(e)}")


def main():
    """主程序入口。"""
    tracker = ExpenseTracker()

    while True:
        print("\n=== 个人记账系统 ===")
        print("1. 记录支出")
        print("2. 查看月度报告")
        print("3. 查看年度报告")
        print("4. 退出")

        try:
            choice = input("\n请选择功能 (1-4): ").strip()

            if choice == '1':
                expense_input = input("请输入支出 (格式:YYYY/MM/DD，描述-金额): ")
                result = tracker.parse_input(expense_input)
                if result:
                    date, desc, amount, category = result
                    tracker.add_expense(date, desc, amount, category)

            elif choice == '2':
                month = input("请输入月份 (格式:YYYY-MM): ").strip()
                if not re.match(r'^\d{4}-(?:0[1-9]|1[0-2])$', month):
                    print("月份格式错误，请使用YYYY-MM 格式")
                    continue
                tracker.generate_monthly_report(month)

            elif choice == '3':
                year = input("请输入年份: ").strip()
                if not re.match(r'^\d{4}$', year):
                    print("年份格式错误，请输入四位数年份")
                    continue
                tracker.generate_yearly_report(year)

            elif choice == '4':
                print("感谢使用！")
                break
            else:
                print("无效选择，请重试")

        except KeyboardInterrupt:
            print("\n程序已中断")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            continue


if __name__ == "__main__":
    main()
