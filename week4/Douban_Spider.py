#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：douban.py 
@File    ：Douban_Spider.py
@Author  ：wkml5994
@Date    ：2021/4/20 19:37
"""
from lxml import etree
import requests
import re
import pandas as pd
import time
import random
import pymysql
from fake_useragent import UserAgent
import logging


class DouBan:
    def __init__(self):
        self.url = "https://movie.douban.com/top250?start="
        ua = UserAgent(verify_ssl=False, path='fake_useragent.json')
        self.headers = {
            "User-Agent": ua.random,
            "Cookie": 'bid=DUQLFahY7bk; __yadk_uid=WHzOfcwOxfwGIRWqClOS6DyxTP8emD6L; __gads=ID=85482c7052673ff5-22c30af575c70098:T=1618828908:RT=1618828908:S=ALNI_MaEJ-SyoCWQA7YErcSJYXj-51IMrQ; ll="108306"; _vwo_uuid_v2=D2657FE845AE168D3885B5F49C6247FC7|06e1b18676a4720f3944aa7bbe6d1fb0; __utmz=30149280.1618885836.5.2.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmz=223695111.1618885836.5.2.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; push_doumail_num=0; push_noty_num=0; __utmc=30149280; __utmc=223695111; ap_v=0,6.0; __utma=30149280.1088850650.1618828907.1619022445.1619024957.10; __utmb=30149280.0.10.1619024957; __utma=223695111.201988307.1618828907.1619022445.1619024957.10; __utmb=223695111.0.10.1619024957; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1619024957%2C%22https%3A%2F%2Fcn.bing.com%2F%22%5D; _pk_ses.100001.4cf6=*; dbcl2="236624132:zif6C5Tga68"; ck=G3yo; _pk_id.100001.4cf6=230ff8aea2342249.1618828906.10.1619025306.1619023077.'
        }

    def run(self):
        """
        :return: the list of returned crawls
        """
        infomation = []
        try:
            # 设置翻页
            for i in range(0, 10):
                response = requests.get(self.url + str(i * 25), headers=self.headers, )
                if response.status_code == 200:
                    content = response.content.decode('utf8')
                    html = etree.HTML(content)
                    urls = html.xpath("//div[@class='hd']//a/@href")
                    # 每页抓25条
                    for url in urls:
                        time.sleep(random.random())
                        # 请求
                        response = requests.get(url, headers=self.headers)
                        # 网页字符串
                        content = response.content.decode('utf8')
                        # 转化为html格式
                        html = etree.HTML(content)

                        title = html.xpath('//div//h1/span[@property]/text()')[0] if \
                            html.xpath('//div//h1/span[@property]/text()') else None
                        director = html.xpath('//div//span/a[@rel="v:directedBy"]/text()')[0] if \
                            html.xpath('//div//span/a[@rel="v:directedBy"]/text()') else None
                        # 日期带着括号用xpath后续需要再处理
                        release_year = re.findall('class="year">.(.*?).</span>', content, re.S)[0] if \
                            re.findall('class="year">.(.*?).</span>', content, re.S) else None
                        release_time = html.xpath('//div//span[@property="v:initialReleaseDate"][1]/text()')[0] if \
                            html.xpath('//div//span[@property="v:initialReleaseDate"][1]/text()') else None
                        # 评星在字符串里用xpath不好抓
                        star = float(re.findall('"ll bigstar bigstar(.*?)">', content, re.S)[0]) / 10 if \
                            re.findall('"ll bigstar bigstar(.*?)">', content, re.S) else None
                        comment = html.xpath('//div/a[@href="comments"]/span/text()')[0]
                        # 简介有分段、被隐藏
                        introduction = []
                        # 如果被隐藏的话
                        if html.xpath('//div//span[@class="all hidden"]'):
                            # 分段抓取再合并
                            for j in range(len(html.xpath('//div//span[@class="all hidden"]/text()'))):
                                introduction.append(
                                    html.xpath('//div//span[@class="all hidden"]/text()[{}]'.format(j + 1))[0].strip())
                        # 如果没有被隐藏
                        else:
                            for j in range(len(html.xpath('//div//span[@property="v:summary"]/text()'))):
                                introduction.append(
                                    html.xpath('//div//span[@property="v:summary"]/text()[{}]'.format(j + 1))[
                                        0].strip())
                        introduction = ''.join(introduction)

                        info = {
                            'title': title,
                            'director': director,
                            'release_year': release_year,
                            'release_time': release_time,
                            'star': star,
                            'comment': int(comment),
                            'introduction': introduction,
                        }
                        print(info)
                        infomation.append(info)
        except Exception as e:
            logging.exception(e)
        return infomation


def make_table_sql(df):
    """
    :param df: DataFrame of list
    :return: the sql statements
    """
    columns = df.columns.tolist()
    types = df.dtypes
    # 添加id 制动递增主键模式
    make_table = []
    for item in columns:
        if 'int' in str(types[item]):
            sql_char = item + ' INT'
        elif 'float' in str(types[item]):
            sql_char = item + ' FLOAT'
        elif item == 'introduction':
            sql_char = item + ' TEXT'
        elif 'object' in str(types[item]):
            sql_char = item + ' VARCHAR(255)'
        elif 'datetime' in str(types[item]):
            sql_char = item + ' DATETIME'
        make_table.append(sql_char)
    return ','.join(make_table)


def df_to_mysql(db_name, table_name, df):
    """
    :param db_name: the name of database
    :param table_name: the name of table
    :param df: The DataFrame to be stored in the sql
    :return:
    """
    # 创建database和table
    cursor.execute('CREATE DATABASE IF NOT EXISTS {}'.format(db_name))
    # 选择连接database
    conn.select_db(db_name)
    cursor.execute('DROP TABLE IF EXISTS {}'.format(table_name))
    cursor.execute('CREATE TABLE {}({})'.format(table_name, make_table_sql(df)))
    values = df.values.tolist()
    # 根据columns个数
    s = ','.join(['%s' for _ in range(len(df.columns))])
    # executemany批量操作
    try:
        cursor.executemany('INSERT INTO {} VALUES ({})'.format(table_name, s), values)
    except Exception as e:
        logging.exception(e)
        # 发生错误则回滚
        conn.rollback()


if __name__ == '__main__':
    # 连接数据库配置
    conn = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           password='159789456',
                           charset='utf8')
    cursor = conn.cursor()
    # 自动提交
    conn.autocommit(1)

    # 实例化
    spider = DouBan()
    doubanTop = spider.run()
    # 至少爬到一条数据
    try:
        if doubanTop[0]:
            doubanTop = pd.DataFrame(doubanTop, index=range(1, len(doubanTop) + 1))
            doubanTop.insert(0, 'top', range(1, len(doubanTop) + 1))
            df_to_mysql('douban', 'top_info', doubanTop)
            doubanTop.to_csv('C:/Users/wkml996/Desktop/douban/douban.csv', encoding='utf-8-sig')
    except Exception as e:
        logging.exception(e)
