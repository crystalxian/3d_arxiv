import sqlite3

# 连接数据库并查看表结构
def check_db():
    try:
        conn = sqlite3.connect('3dgen_papers.db')  # 确认数据库文件名
        cursor = conn.cursor()
        
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("数据库中的表:", tables)
        
        # 检查papers表中的数据
        cursor.execute("SELECT COUNT(*) FROM papers;")
        count = cursor.fetchone()[0]
        print(f"papers表中有 {count} 条记录")
        
        # 查看一些示例数据
        cursor.execute("SELECT id, title, published FROM papers LIMIT 5;")
        samples = cursor.fetchall()
        print("示例数据:", samples)
        
        conn.close()
    except Exception as e:
        print(f"数据库检查出错: {e}")