import time


class Timers:
    timer = {}

    def start_timer(self, name, count_towards_total):
        # Overlapping timers should use count_towards_total=False
        if name in self.timer:
            raise Exception(f'Using repeated timer: {name}')

        self.timer[name] = {
            'start': time.perf_counter(),
            'count_towards_total': count_towards_total,
            'end': 0,
            'duration': 0,
            'completed': False,
        }

    def end_timer(self, name):
        if name not in self.timer:
            raise Exception('Using not existent timer')

        self.timer[name]['end'] = time.perf_counter()
        self.timer[name]['duration'] = self.timer[name]['end'] - self.timer[name]['start']
        self.timer[name]['completed'] = True

    def get_timer_duration(self, name):
        if self.timer[name]['completed']:
            return self.timer[name]['duration']

        raise Exception(f'Timer {name} not completed')

    def get_timers(self, total_duration):
        total_measured = 0

        values_to_log = [('Total', total_duration)]

        for name, item in self.timer.items():
            if item['count_towards_total'] and item['completed']:
                total_measured += item['duration']

            x = 'T' if item['count_towards_total'] else 'F'

            if item['completed']:
                values_to_log.append((name + f' ({x})', item['duration']))
            else:
                values_to_log.append((name + f' ({x})', None))

        missing = total_duration - total_measured
        values_to_log.append(('Missing', missing))

        log_parts = []
        for name, value in values_to_log:
            if value is None:
                log_parts.append(f'{name}: None')
            else:
                log_parts.append(f'{name}: {value:.1f}s')

        return 'Timers: ' + ' | '.join(log_parts)
