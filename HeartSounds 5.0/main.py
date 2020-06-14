'''
@brief Script that opens a serial port and plots incoming data

@author: 
@date: 
'''

# Import modules
import serial
import serial.tools.list_ports
import struct
import threading
import time
import random
import statistics
import numpy as np
from scipy import signal  
import os
from multiprocessing import Process,Pipe
import datetime

'''
#import modules for signal processing
from scipy.signal import hilbert
import numpy as np
'''
# Import kivy-related modules
from kivy.app import App
from kivy.clock import Clock
from kivy.garden.graph import LinePlot, MeshStemPlot  # pylint: disable=no-name-in-module,import-error
from kivy.properties import ObjectProperty, BooleanProperty, NumericProperty, ListProperty, StringProperty  # pylint: disable=no-name-in-module
from kivy.uix.boxlayout import BoxLayout
from kivy.event import EventDispatcher
from kivy.uix.behaviors import ButtonBehavior  
from kivy.uix.image import Image
from kivy.animation import Animation 
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder

import matplotlib.pyplot as plt


'''
@brief Class that handles serial communication and data parsing
'''
check_thread = 0

class CustomSerial(EventDispatcher):
    # EventDispatcher
    # Boolean to determine when device is connected
    is_connected = BooleanProperty(False)
    sample_data = NumericProperty(0)  # Numeric property with sampled data
    sample_data_2 = NumericProperty(0)  # Numeric property with sampled data
    flag_update=BooleanProperty(False)
    C_freq = NumericProperty(0)
    file_name = StringProperty('default')
    
    def __init__(self, **kwargs):
        self.port_name = ""         # Name of the port
        self.ser = serial.Serial()  # Serial object
        self.is_streaming = False   # Streaming boolean flag
        self.read_state = 0         # Serial parser state machine flag

        self.n_sample = 0
        self.elapsed_time = 0
        self.time_check = 0
        self.count_check = 0
        self.samples_tot = 0

        global check_thread
        check_thread = check_thread + 1

        now = datetime.datetime.now()
        self.time_stamp = "%d%s%s_%s%s%s" % (
            now.year, str(now.month) if now.month >= 10 else '0' +
            str(now.month),
            str(now.day) if now.day >= 10 else '0' + str(now.day), str(now.hour) if (now.hour) >= 10
            else '0' + str(now.hour), str(now.minute) if (now.minute) >= 10
            else '0' + str(now.minute), str(now.second) if (now.second) >= 10
            else '0' + str(now.second))
        if not os.path.exists("Data"):
            os.makedirs("Data")
        self.file_name = os.path.join("Data", self.time_stamp + '_amplitude_values.txt')
        self.f = open(self.file_name, 'a')    
    

        '''
        Start a thread to connect to the board.
        This allows to check the connected COM ports without freezing the UI.
        '''
        if(check_thread == 1):
            self.connection_thread = threading.Thread(target=self.find_port)  
            self.connection_thread.daemon = True
            self.connection_thread.start()
        else:
            pass

    def connect(self):
        '''
        @brief Connect to a serial port
        '''
        try:
            self.ser = serial.Serial(
                port=self.port_name, baudrate=115200, write_timeout=0, timeout=2)
            if (self.ser.is_open):
                self.is_connected = True
                return True
        except serial.SerialException:
            return False
        except ValueError:
            return False

    def find_port(self):
        '''
        @brief Loop over all the available serial ports to find the right one.
        '''
        found_board = False
        while (not found_board):
            # Get the list of the ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                # Get the name of the port and check if port is the right one
                name = port.device
                found_board = self.check_port(name)
                if (found_board):
                    # If we've found it, connect to it
                    print('Serial port found: %s' % name)
                    self.port_name = name
                    connected = self.connect()
                    if (connected == True):
                        break
                    else:
                        found_board = False
                time.sleep(1)


    def check_port(self, name):
        '''
        @brief Check if port sends expected string when prompted.

        When we send a char 'v', the board should reply with "HeartSounds Project$$$".
        Here we send a 'v' and check if "HeartSounds" is present in the received string.
        '''
        print('Checking port: %s' % name)
        try:
            ser = serial.Serial(port=name, baudrate=115200,
                                write_timeout=0, timeout=0)
            if (ser.is_open):
                
                ser.write('v'.encode('utf-8'))
                
                time.sleep(1)
                line = ''
                c = ''
                char_counter = 50
                #
                if (ser.inWaiting()):
                    while('$$$' not in line):
                        c = ser.read().decode('utf-8', errors='replace')
                        line += c
                        char_counter -= 1
                        if (char_counter == 0):
                            break
                ser.close()
                if ('HeartSounds' in line):
                    return True
                else:
                    return False
            return False
        except serial.SerialException:
            return False
        except ValueError:
            return False

    def start_streaming(self):
        '''
        @brief Start data streaming and set up thread for data parsing.
        '''
        
        self.ser.write('b'.encode('utf-8'))
        self.is_streaming = True
        self.read_state = 0
        self.count_check = 0
    
        streaming_thread = threading.Thread(target=self.read_data)
        streaming_thread.daemon = True
        streaming_thread.start()
        

    def read_data(self):
        '''
        @brief Read data while streaming is active
        '''
        while self.is_streaming:
            try:
                data, data2 = self.parse_data()
                self.count_check = 0
            except:
                data = None
                data2 = None
                self.count_check = self.count_check + 1
                if(self.count_check > 10000):
                    self.is_streaming = False
                    self.count_check = 0
                    print('Error in self.parse_data()')

            # Check if received data is not None (it may happen when we stop streaming)
            if(not((data == None) or (data2 == None))):
                self.sample_data = data
                self.sample_data_2 = data2
                self.f.write(str(self.sample_data) + '\n')
                self.f.write(str(self.sample_data_2) + '\n')
                self.samples_tot = self.samples_tot + 2
                self.flag_update=not(self.flag_update)
 
    def Get_freq(self):

        if (self.time_check== 0):
            self.ref_time = time.time()
            self.time_check = 1
            self.n_sample = 0

        self.n_sample = self.n_sample +2
        self.elapsed_time = time.time() - self.ref_time

        if(self.elapsed_time > 3 and (self.n_sample!=0)):
            self.C_freq = self.n_sample/3
            self.ref_time = time.time()
            self.n_sample = 0
            print('Sampling frequency:',self.C_freq,'Hz')
        else:
            pass

    def parse_data(self, max_bytes_to_skip=3000):
        '''
        @brief Parse incoming data
        '''
        def read(n):
            '''
            @brief Helper function to read data from serial port
            '''
            bb = self.ser.read(n)
            if not bb:
                print("Device appears to be stalled.. Quitting")
                return -999
            else:
                return bb
        # Serial parser state machine
        # ---------Start Byte --------
        for rep in range(max_bytes_to_skip):
            if self.read_state == 0:
                
                b = read(1)
                
                if (b == -999):
                    break
                if (struct.unpack('B', b)[0] == 0xA0):
                    self.read_state = 1
                    
                    
            # --------- Data ---------
            elif self.read_state == 1:
                b = read(2)
                data = struct.unpack('2B', b)
                sample = data[0] << 8 | data[1]
                self.read_state = 2
                
            elif self.read_state == 2:
                b = read(2)
                data = struct.unpack('2B', b)
                sample2 = data[0] << 8 | data[1]
                self.read_state = 3
                
            # ---------End Byte --------
            elif self.read_state == 3:
                b = read(1)
                if (struct.unpack('B', b)[0] == 0xC0):
                    
                    self.read_state = 0
                    self.Get_freq()
                    return sample,sample2


    def stop_streaming(self):
        '''
        @brief Stop data streaming
        
        '''
        self.is_streaming = False
        self.ser.write('s'.encode('utf-8'))
       
class MyPlotScreen(Screen):
    '''
    @brief Container widget for UI elements
    '''
    graph = ObjectProperty(None)
    start = ObjectProperty(None)
    hrate = ObjectProperty(0)
    labelSampfreq = ObjectProperty(None)
    button_color_start = BooleanProperty(False)
    button_color_stop = BooleanProperty(False)
    stop = ObjectProperty(None)
    isShownMenu = BooleanProperty(False)
    isShownPlotSettings = BooleanProperty(False)
    isShownHR = BooleanProperty(False)
    inputmin = NumericProperty(0)
    inputmax = NumericProperty(0)
    start_rec = ObjectProperty(0)
    stop_rec = ObjectProperty(0)
    go_report = ObjectProperty(0)
    connection_label = ObjectProperty(None)
    report_data = ListProperty([])
    HR_corrected = ListProperty([])
    peaks_positions = ListProperty([])
    first_sample = NumericProperty(0)
    close_file = BooleanProperty(False)

    def __init__(self,**kwargs):

        super(MyPlotScreen, self).__init__()
        # Set up custom serial object and bind its properties
        self.ser = CustomSerial()
        self.ser.bind(is_connected=self.connected)
        self.ser.bind(flag_update=self.update_plot)
        self.ser.bind(C_freq=self.Show_SampFreq)
        self.bind(button_color_start=self.change_color_start)
        self.bind(button_color_stop=self.change_color_stop)
        self.bind(close_file=self.close_files)
        self.report_flag = 0
        self.HP_low = 0

    def change_color_start(self, instance, value):

        if (value == True):
            self.start_rec.text = "Recording"
            self.start_rec.color = 0, 1, 0, 1
            self.report_flag = 1
            self.first_sample = self.ser.samples_tot - 3000
        else:
            self.start_rec.text = "Start\nrecording"
            self.start_rec.color = 1, 1, 1, 1 

    def change_color_stop (self, instance, value):

        if (value == True):
            self.start_rec.text = "Start\nrecording"
            self.start_rec.color = 1, 1, 1, 1 
            self.go_report.disabled = False
            self.report_flag = 0
        else:
            pass

    def connected(self, instance, value):
        '''
        @brief Callback called when the ser.is_connected property changes
        '''
        if (value == True):
            self.start.disabled = False
            self.stop.disabled = False
            self.connection_label.text = "Connected"
            self.connection_label.color = 0, 1, 0, 1
        else:
            pass

    def on_graph(self, instance, value):
        '''
        @brief Callback called when graph object is ready on the UI
        '''
        
        self.graph.label_options['color'] = 0, 0, 0, 1
        self.graph.label_options['bold'] = True
        self.graph.xlabel = "Time [s]"
        self.graph.ylabel = "Amplitude [V]"
        self.graph.x_ticks_major = 5  # x ticks major value
        self.graph.x_ticks_minor = 1  # x ticks minor value
        self.graph.x_grid_label = True  # grid on x axis

        self.n_seconds = 3
        self.graph.y_ticks_major = 1  # x ticks major value
        self.graph.y_ticks_minor = 1  # x ticks minor value
        self.graph.y_grid_label = True  # grid on x axis
        self.graph.xmin = - self.n_seconds
        self.graph.xmax = 0
        self.graph.ymin = 0
        self.graph.ymax = 5

        self.sample_rate = 1000  # 10 samples per second
        self.n_points = self.n_seconds * self.sample_rate
        #self.n_seconds * self.sample_rate  # Number of points to plot
        #  Distance between points
        self.time_between_points = (self.n_seconds)/float(self.n_points)

        self.plot = LinePlot(color=[0, 0, 0, 1.0])
        self.plot.line_width = 1
        # Populate lists of points
        self.x_points = [x for x in range(-self.n_points, 0)]
        self.y_points = [0 for y in range(-self.n_points, 0)]
        for j in range(self.n_points):
            self.x_points[j] = -self.n_seconds + j * self.time_between_points
        # Zip them and add them to the plot points
        self.plot.points = zip(self.x_points, self.y_points)

        self.graph.add_plot(self.plot)
        

    def Show_SampFreq(self, instance, value):
        try:
            self.labelSampfreq.text = str(round(self.ser.C_freq,1)) + ' Hz'
        except: 
            if self.ser.C_freq == None:
                self.labelSampfreq.text = "NoValue"

    def on_start(self, instance, value):

        self.start.bind(on_release=self.start_streaming)

    def on_stop(self, instance, value):

        self.stop.bind(on_release=self.stop_streaming)
    
    def start_streaming(self, instance):
        '''
        @brief Start streaming data
        '''
        self.HR_init()
        self.update_hr()
        self.ser.start_streaming()

    def stop_streaming(self, instance):
        '''
        @brief Stop streaming data
        '''
        self.ser.stop_streaming()
    
    def update_hr(self):
        '''
        a function will compute the actual value of th heart rate called self.heart_rate and
        it will be displayed as the text attribute of the label with id:_heart_rate (hr) now i set it as 75 in init of Myplot class
        '''
        if(self.isShownHR == 1):
            self.hrate.text = str(int(self.HR_corrected[-1]))
        else:
            self.hrate.text = 'calibration'  # not visible as there is a flag in HR visibility equal to 0

    def update_plot(self, instance, value):
        '''
        @brief Update plot with new data point.
        '''
        self.y_points.append(self.y_points.pop(0))
        self.y_points[-1] = self.ser.sample_data/1000      # y in 0-5 range while incoming data in 0-5000
        self.y_points.append(self.y_points.pop(0))
        self.y_points[-1] = self.ser.sample_data_2/1000
        self.plot.points = zip(self.x_points, self.y_points)

        if(self.report_flag == 1):
            self.report_data.append(self.ser.sample_data/1000)
            self.report_data.append(self.ser.sample_data_2/1000)

        if((self.ser.samples_tot-self.HP_low) >= 3000):
            self.start_computing(self.y_points)
        
    def HR_init(self):

        self.time_check = 0          # at the beginning of the program execution
        self.first_time = True       # first HR computation
        self.heart_period = 0.5      
        self.fs = 1000
        self.heartrate = []
        self.n_HR_check = 5
        self.zero_time = time.time()  

        # definition of the filenames
        now = datetime.datetime.now()
        self.time_stamp = "%d%s%s_%s%s%s" % (
            now.year, str(now.month) if now.month >= 10 else '0' +
            str(now.month),
            str(now.day) if now.day >= 10 else '0' + str(now.day), str(now.hour) if (now.hour) >= 10
            else '0' + str(now.hour), str(now.minute) if (now.minute) >= 10
            else '0' + str(now.minute), str(now.second) if (now.second) >= 10
            else '0' + str(now.second))
        self.file_name = os.path.join("Data", self.time_stamp + '_HR_values.txt')
        self.file_name_corrected = os.path.join("Data", self.time_stamp + '_HR_corrected_values.txt')
        self.file_name_peaks = os.path.join("Data", self.time_stamp + '_peaks_positions.txt')     
       
    def start_computing(self,y_points):

        # at the beginning of the HR computation we read the number of data we received so far
        self.n_samples = self.ser.samples_tot
            
        # we define the initial time of the actual 3s window we're using for the computation of the HR       
        if(not(self.zero_time == None)):
            self.data_start = time.time() - self.zero_time - 3  # initial time of the 3s window
            print('Computation at t:', time.time() - self.zero_time)
        

        # remove the mean value
        self.data_extracted = y_points[:]
        self.mean=statistics.mean(self.data_extracted)
        self.data_extracted=[i-self.mean for i in self.data_extracted]

        # filter the signal 25-400 Hz bandpass filter
        self.ws=np.array([10,150])/(self.fs/2) # 25-400 mic
        self.b, self.a = signal.butter(1, self.ws , btype='bandpass', analog=False) 
        self.actual_data = signal.filtfilt(self.b, self.a, np.array(self.data_extracted))

        # design low pass filter at 8 Hz useful for the envelope
        self.ws2=8/(self.fs/2) 
        self.b2, self.a2 = signal.butter(1, self.ws2 , btype='lowpass', analog=False)

        # computing the envelope and remove its mean
        self.homomorphic_envelope = np.exp(signal.filtfilt(self.b2, self.a2, np.log(np.abs(signal.hilbert(self.actual_data)))))
        self.mean_envelope=statistics.mean(self.homomorphic_envelope)
        self.homomorphic_envelope = [i-self.mean_envelope for i in self.homomorphic_envelope]

        # compute the autocorrelation of the envelope and normalize it by its variance
        self.signal_autocorrelation=np.correlate(self.homomorphic_envelope,self.homomorphic_envelope, mode='full')
        self.signal_autocorrelation = self.signal_autocorrelation[len(self.homomorphic_envelope)+1:]
        self.normalized_autocorrelation = [x/self.signal_autocorrelation[0] for x in self.signal_autocorrelation]

        # definition of the time window in which we want to find the max autocorrelation
        self.min_index=round(0.272*self.fs)
        self.max_index=round(3*self.fs)

        # find the max autocorrelation, its position is the value of the HP
        self.position = np.where(self.normalized_autocorrelation[self.min_index:self.max_index] == np.amax(self.normalized_autocorrelation[self.min_index:self.max_index]))
        self.position=self.position[0]
        self.max_position=self.position + self.min_index - 1
        self.heart_period = self.max_position/self.fs
        self.HP_low = self.HP_low + self.max_position # send the HP to the "check_time" process for the elapsed time control
        self.heartrate.append(60/(self.max_position/self.fs))

        print('HR equal to:',self.heartrate[-1])

        # write values in a txt file
        with open(self.file_name, 'a') as f:
            f.write(str(float(self.heartrate[-1])) + '\n')
        
        if(self.isShownHR == 1):
            # Here we control the current HR value
            # if btw 2 consecutive beats we've a difference in HR values higher then 20 bpm we consider the new value as an error.
            # In addition if the max value in the window 1:HP is lower then 0.1 it means that the signal we're acquiring is not correct
            # so we avoid these HR values
            if(np.absolute(self.heartrate[-1]-self.HR_corrected[-1]) <= 20 and np.amax(self.actual_data[1:int(self.heart_period*self.fs)]) > 0.01): # 0.03 mic

                first_max = np.where(self.actual_data[1:int(self.heart_period*self.fs)] == np.amax(self.actual_data[1:int(self.heart_period*self.fs)]))
                # print(np.amax(self.actual_data[1:int(self.heart_period*self.fs)]))
                first_max=first_max[0]  # here first_max[0] take the numbers (we remove the variable type from first_max)
                # we control that 2 consecutive peaks have a distance at least higher then 0.272s (220 bpm)
                if(((first_max[0] + self.n_samples - 3*self.fs)-self.peaks_positions[-1]) >= 0.272*self.fs):
                    self.peaks_positions.append((self.n_samples - 3*self.fs + first_max[0])) # now first_max[0] take only its first value
                    with open(self.file_name_peaks, 'a') as k:
                        k.write(str(float(self.peaks_positions[-1])) + '\n')

                # print('actual:',self.heartrate[-1])
                # print('previous:',self.HR_corrected[-1])
                self.HR_corrected.append(self.heartrate[-1])
                self.update_hr()

                with open(self.file_name_corrected, 'a') as g:
                    g.write(str(float(self.HR_corrected[-1])) + '\n')
            else:
                print('HR error detected')

        # we check that the system is stable and HR computed are sufficiently coherent
        if(len(self.heartrate) >= self.n_HR_check and self.isShownHR == 0):
            if(np.amax(self.heartrate[-self.n_HR_check:])-np.amin(self.heartrate[-self.n_HR_check:]) <= 10):
                # if in a 10-elements array the difference btw the max and min value
                # is lower then 10 bpm, HRs are regular  
                print('Regular beats')
                self.HR_corrected = self.heartrate[-self.n_HR_check:] # we save these 10 regular HRs and write in a file
                with open(self.file_name_corrected, 'a') as g:
                    for HR in self.HR_corrected:
                        g.write(str(float(HR)) + '\n')

                # compute the peak position considering the max value of the signal in a window 1:HP
                first_max = np.where(self.actual_data[1:int(self.heart_period*self.fs)] == np.amax(self.actual_data[1:int(self.heart_period*self.fs)])) 
                # print(np.amax(self.actual_data[1:int(self.heart_period*self.fs)]))
                first_max = first_max[0]  # here first_max[0] take the numbers (we remove the variable type from first_max)
                self.peaks_positions.append((self.n_samples - 3*self.fs + first_max[0]))  # now first_max[0] take only its first value
                with open(self.file_name_peaks, 'a') as k:
                    k.write(str(float(self.peaks_positions[-1])) + '\n')
                print('First start: ',self.data_start)
                # print(self.actual_data[-20:])
                self.isShownHR = 1
                self.update_hr()

    def close_files(self, instance, value):
        self.ser.f.close
        #self.k.close
        
class ReportScreen(Screen):
    
    '''
    @brief Container widget for UI elements
    '''
    graph_hr = ObjectProperty(None)
    graph_signal = ObjectProperty(None)
    plot_graph = BooleanProperty(False)
    showvalidation = BooleanProperty(False)
    f_min = ObjectProperty(0)
    f_max = ObjectProperty(0)
    f_mean = ObjectProperty(0)

    def __init__(self,**kwargs):
        super(ReportScreen, self).__init__()
        self.bind(showvalidation = self.show_peaks)

        self.fs=995
        self.ws=np.array([10,150])/(self.fs/2) # 25-400 mic
        self.b, self.a = signal.butter(1, self.ws , btype='bandpass', analog=False)

    def on_graph_signal(self, instance, value):

        self.graph_signal.label_options['color'] = 0, 0, 0, 1
        self.graph_signal.label_options['bold'] = True
        self.graph_signal.xlabel = "Time [s]"
        self.graph_signal.ylabel = "Amplitude [V]"
        self.graph_signal.x_ticks_major = 5  # x ticks major value
        self.graph_signal.x_ticks_minor = 1  # x ticks minor value
        self.graph_signal.x_grid_label = True  # grid on x axis
        self.graph_signal.y_ticks_major = 0.5  # y ticks major value
        self.graph_signal.y_ticks_minor = 0.5  # y ticks minor value
        self.graph_signal.y_grid_label = True  # grid on x axis
        self.sample_rate = 995  # 1000 samples per second
        self.plot = LinePlot(color=[0, 0, 0, 1.0])
        self.plot.line_width = 1
        self.plot2 = MeshStemPlot(color=[1, 0, 0, 1.0])        
    
    def on_graph_hr(self, instance, value):

        self.graph_hr.label_options['color'] = 0, 0, 0, 1
        self.graph_hr.label_options['bold'] = True
        self.graph_hr.xlabel = "Time [s]"
        self.graph_hr.ylabel = "Heart Rate [bpm]"
        self.graph_hr.x_ticks_major = 5  # x ticks major value
        self.graph_hr.x_ticks_minor = 1  # x ticks minor value
        self.graph_hr.x_grid_label = True  # grid on x axis
        self.graph_hr.y_ticks_major = 10  # x ticks major value
        self.graph_hr.y_ticks_minor = 5  # x ticks minor value
        self.graph_hr.y_grid_label = True  # grid on y axis
        self.plot_hr = LinePlot(color=[0, 0, 0, 1.0])
        self.plot_hr.line_width = 1

    def on_enter(self):
        
        self.amplitude_data = self.manager.get_screen('myplot').report_data
        self.peaks = self.manager.get_screen('myplot').peaks_positions
        self.peak_check = self.manager.get_screen('myplot').first_sample
        self.HR_data = self.manager.get_screen('myplot').HR_corrected


        self.mean_report = statistics.mean(self.amplitude_data)
        self.amplitude_data=[i-self.mean_report for i in self.amplitude_data]        
        self.y_data = signal.filtfilt(self.b, self.a, np.array(self.amplitude_data))
        self.y_data=[i+self.mean_report for i in self.y_data]

        self.n_points = len(self.amplitude_data)
        self.n_seconds = round(self.n_points / self.sample_rate,1)
        self.time_between_points = 1/self.sample_rate
        self.x_points = [x for x in range(self.n_points)]
        for j in range(self.n_points):
            self.x_points[j] = j * self.time_between_points

        self.graph_signal.xmin = 0 
        self.graph_signal.xmax = self.n_seconds
        self.graph_signal.ymin = float(round(0.99*np.amin(self.y_data),1))
        self.graph_signal.ymax = float(round(1.01*np.amax(self.y_data),1))

        self.plot.points = zip(self.x_points, self.y_data)        
        self.graph_signal.add_plot(self.plot)
        
        # Update the HR plot
        self.peaks_corrected = [i-self.peak_check for i in self.peaks if i >= self.peak_check]
        step = len(self.peaks)-len(self.peaks_corrected)
        self.x_points_hr = [i/self.fs for i in self.peaks_corrected]
        self.graph_hr.xmin = 0
        self.graph_hr.xmax = float(round(np.amax(self.x_points_hr),1))
        self.graph_hr.ymin = float(round(0.9*np.amin(self.HR_data)))
        self.graph_hr.ymax = float(round(1.1*np.amax(self.HR_data)))

        self.plot_hr.points = zip(self.x_points_hr, self.HR_data[5+step:])        
        self.graph_hr.add_plot(self.plot_hr)

        self.f_min.text = 'Frequency MIN:' + str(int(np.amin(self.HR_data)))
        self.f_max.text = 'Frequency MAX:' + str(int(np.amax(self.HR_data)))
        self.f_mean.text = 'Frequency MEAN:' + str(int(np.mean(self.HR_data)))
        
    def show_peaks(self, instance, value):
        self.file_name_data = self.manager.get_screen('myplot').ser.file_name
        self.file_name_peaks = self.manager.get_screen('myplot').file_name_peaks
        self.file_name_data = self.file_name_data[5:]
        self.file_name_peaks = self.file_name_peaks[5:]
        
        f = open("C:/Users/massi/Data/" + self.file_name_data, "r")
        data=f.read()
        data_splitted=data.split('\n')
        results = [float(i) for i in data_splitted[:-1]]
        f.close

        signal_data=[i/1000 for i in results]
        mean=statistics.mean(results)
        signal_data=[i-mean for i in signal_data]

        fs = 995
        ws=np.array([10,150])/(fs/2)
        b, a = signal.butter(5, ws , btype='bandpass', analog=False)
        data_filtered = signal.filtfilt(b, a, np.array(signal_data))
        
        f = open("C:/Users/massi/Data/" + self.file_name_peaks, "r")
        data=f.read()
        data_splitted=data.split('\n')
        peaks = [int(float(i)) for i in data_splitted[:-1]]
        y_peaks = [1.2*data_filtered[i] for i in peaks]
        y_max = np.amax(y_peaks)
        peaks = [i/fs for i in peaks]
        f.close

        plt.figure(1)
        plt.plot(np.linspace(0,len(data_filtered)/fs,len(data_filtered)), data_filtered,Linewidth = 0.5)
        plt.stem(peaks, y_max*np.ones(len(peaks)),linefmt='red',markerfmt= 'rx',basefmt= 'blue',use_line_collection=True)
        plt.title('Signal with peaks')
        plt.xlabel('Time[s]')
        plt.ylabel('Amplitude [V]')

        plt.show()

        
    
class WindowManager(ScreenManager):
    pass


'''
class created to allow batton with icons 
'''
class ImageButton(ButtonBehavior, Image):  
    def on_press(self):  
        print('pressed')

class RealTimePlotApp(App):
    def build(self):
        return kv

if __name__ == '__main__':
    kv = Builder.load_file("realtimeplot.kv")
    RealTimePlotApp().run()
    