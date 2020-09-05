import numpy as np
import statistics as stat


class Request:
    def __init__(self, created_time, processing_time):
        self.created_time = created_time
        self.processing_time = processing_time
        self.end_time = 0


class Queue:

    def __init__(self, m):
        self.queue_size = m
        self.requests = []

    def add_request(self, request):
        if len(self.requests) < self.queue_size:
            self.requests.append(request)

    def pop_request(self):
        if len(self.requests) != 0:
            request = self.requests[0]
            self.requests.remove(request)
            return request

    def is_available(self):
        return False if len(self.requests) == self.queue_size else True

    def is_empty(self):
        return len(self.requests) == 0


class ChannelSystem:

    def __init__(self, n):
        self.channels = [Channel() for _ in range(n)]

    def get_any_free_channel(self):
        for channel in self.channels:
            if not channel.is_busy:
                return channel

    def get_busy_channels_count(self):
        requests = 0
        for channel in self.channels:
            if channel.is_busy:
                requests += 1
        return requests


class Channel:
    def __init__(self):
        self.is_busy = False
        self.current_request = None

    def add_request(self, request, current_time):
        self.current_request = request
        self.is_busy = True
        self.current_request.end_time = current_time + self.current_request.processing_time

    def pop_request(self):
        self.current_request = None
        self.is_busy = False


class SMO:

    def __init__(self, m, n, time_limit, dt, request_intensity, channel_intensity, p):
        self.channel_count = n
        self.queue_len = m
        self.time_limit = time_limit
        self.dt = dt
        self.p = p
        self.request_intensity = request_intensity
        self.channel_intensity = channel_intensity
        self.current_time = 0
        self.new_request = None
        self.new_queue = Queue(m=m)
        self.channel_system = ChannelSystem(n=n)

        self.completed_count = 0
        self.rejected_count = 0
        self.received_count = 0
        self.cancelled_count = 0
        self.all_count = 0

        self.theory_states = [0 for _ in range(m + n + 1)]
        self.practical_states = [0 for _ in range(m + n + 1)]

        self.practical_queue_len = []
        self.practical_busy_channels = []
        self.practical_requests_count = []
        self.all_states = []

    def run(self):
        self.create_request()
        tick_count = 0
        while True:
            tick_count += 1
            self.current_time += self.dt
            if self.current_time >= self.time_limit:
                break
            self.free_channels()
            self.free_queue()
            count = len(self.new_queue.requests) + self.channel_system.get_busy_channels_count()
            self.practical_states[count] += 1
            self.practical_requests_count.append(count)
            self.practical_queue_len.append(len(self.new_queue.requests))
            self.practical_busy_channels.append(self.channel_system.get_busy_channels_count())
            temp_states = [x / sum(self.practical_states) for x in self.practical_states]
            self.all_states.append(temp_states)
            self.try_create_new_request()

        print("practical")
        self.practical_states = [round(x / sum(self.practical_states), 4) for x in self.practical_states]
        print(self.practical_states)
        self.theory_states = self.calculate_theoretical_states_probabilities()
        print("theory")
        print(self.theory_states)

        self.calculate_bandwidth_statistic()
        average_waiting, average_waiting_pract = self.calculate_average_queue_len()
        average_channels = self.calculate_average_busy_channels_count()
        average_system, average_system_pract = self.calculate_average_requests(average_waiting, average_channels)
        self.calculate_average_request_live_in_queue(average_waiting, average_waiting_pract)
        self.calculate_average_request_live_in_system(average_system, average_system_pract)

    def calculate_bandwidth_statistic(self):
        Q_practical = 1 - self.practical_states[-1]
        Q_theory = 1 - self.theory_states[-1]
        A_practical = self.request_intensity * Q_practical
        A_theory = self.request_intensity * Q_theory
        print("Q theoretical: {}, practical: {}".format(Q_theory, Q_practical))
        print("A theoretical: {}, practical: {}".format(A_theory, A_practical))

    def calculate_average_queue_len(self):
        average_practical = stat.mean(self.practical_queue_len)
        average_theory = 0
        for i in range(1, self.queue_len + 1):
            average_theory += i * self.theory_states[self.channel_count + i]
        print("Average queue length theoretical: {}, practical: {}".format(average_theory, average_practical))
        return average_theory, average_practical

    def calculate_average_busy_channels_count(self):
        average_practical = stat.mean(self.practical_busy_channels)
        average_theory = 0
        for i in range(self.channel_count + 1):
            average_theory += i * self.theory_states[i]
        for i in range(1, self.queue_len + 1):
            average_theory += self.channel_count * self.theory_states[self.channel_count + i]
        print("Average busy channels theoretical: {}, practical: {}".format(average_theory, average_practical))
        return average_theory

    def calculate_average_requests(self, average_waiting, average_channels):
        pract_average = stat.mean(self.practical_requests_count)
        theor_average = average_waiting + average_channels
        print("Average requests count in SMO theoretical: {}, practical: {}".format(theor_average, pract_average))
        return theor_average, pract_average

    def calculate_average_request_live_in_queue(self, average_request_queue_count, average_request_queue_count_pract):
        pract_average = average_request_queue_count_pract / self.request_intensity
        theor_average = average_request_queue_count / self.request_intensity
        print("Average requests live time in queue SMO theoretical: {}, practical: {}".format(theor_average, pract_average))

    def calculate_average_request_live_in_system(self, average_request_system_count, average_request_system_count_pract):
        pract_average = average_request_system_count_pract / self.request_intensity
        theor_average = average_request_system_count / self.request_intensity
        print("Average requests live time in SMO theoretical: {}, practical: {}".format(theor_average, pract_average))

    def avg_queue(self):
        return sum([(i + 1) * val for i, val in enumerate(self.theory_states[self.channel_count + 1: self.channel_count + 1 + self.queue_len])])

    def avg_processing(self):
        return sum([(i + 1) * val for i, val in enumerate(self.theory_states[1: self.channel_count + 1])]) +\
               sum([self.channel_count * val for val in self.theory_states[self.channel_count + 1: self.channel_count + 1 + self.queue_len]])

    def is_accepted(self):
        return np.random.choice([True, False], p=[self.p, 1 - self.p])

    def put_in_waiting_queue(self, request):
        if not self.new_queue.is_available():
            self.cancelled_count += 1
            return
        self.new_queue.add_request(request)
        return

    def free_channels(self):
        for channel in self.channel_system.channels:
            if channel.is_busy:
                channel_request = channel.current_request
                if self.current_time >= channel.current_request.end_time:
                    channel.pop_request()
                    if self.is_accepted():
                        self.completed_count += 1
                    else:
                        self.rejected_count += 1
                        if not self.new_queue.is_available():
                            self.cancelled_count += 1
                            return
                        self.new_queue.add_request(channel_request)

    def free_queue(self):
        channel = self.channel_system.get_any_free_channel()
        if channel and not self.new_queue.is_empty():
            request = self.new_queue.pop_request()
            channel.add_request(request, self.current_time)

    def try_create_new_request(self):
        if self.current_time >= self.new_request.created_time:
            self.all_count += 1
            if not self.new_queue.is_available():
                self.cancelled_count += 1
            self.create_request()
            self.new_queue.add_request(self.new_request)

    def create_request(self):
        delta = self.generate_next_exponential(self.request_intensity)
        next_request_time = self.current_time + delta
        processing_time = self.generate_next_exponential(self.channel_intensity)
        new_request = Request(created_time=next_request_time, processing_time=processing_time)
        self.new_request = new_request

    def calculate_theoretical_states_probabilities(self):
        m, n, lmdda, mu, p = self.queue_len, self.channel_count, self.request_intensity, self.channel_intensity, self.p
        if p == 0:
            self.theory_states[-1] = 1
        else:
            A = 0
            for i in range(n + 1):
                A += lmdda ** i / (np.math.factorial(i) * (mu * p) ** i)
            B = 0
            for i in range(n + 1, n + m):
                B += lmdda ** i / (np.math.factorial(n) * n ** (i - n) * (mu * p) ** i)
            C = lmdda ** (n + m) / (np.math.factorial(n) * n ** m * mu ** (n + m) * p ** (n + m - 1))
            self.theory_states[0] = 1 / (A + B + C)
            for i in range(1, n + 1):
                self.theory_states[i] = lmdda ** i / (
                        np.math.factorial(i) * (mu * p) ** i) * self.theory_states[0]
            for i in range(n + 1, n + m):
                self.theory_states[i] = lmdda ** i / (
                        np.math.factorial(n) * n ** (i - n) * (
                        mu * p) ** i) * self.theory_states[0]
            last_prob = lmdda ** (n + m) / (np.math.factorial(n) * n ** m * mu ** (n + m) * p ** (n + m - 1))
            self.theory_states[n + m] = last_prob * self.theory_states[0]

        return self.theory_states

    def generate_next_exponential(self, lamb):
        bsv = np.random.uniform()
        return - 1 / lamb * np.log(1 - bsv)


smo = SMO(m=2, n=2, time_limit=1000, dt=0.1, request_intensity=0.5, channel_intensity=0.5, p=0.5)
smo.run()
