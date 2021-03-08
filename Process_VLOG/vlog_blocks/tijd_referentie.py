#!/usr/bin/python3

import datetime
from .vlog_block_base import Vlog_block_base

class Tijd_referentie (Vlog_block_base):
    """Contains the absolute tijd referentie timestamp of the VRI."""
    def __init__(self):
        """Inializes the object with tijd attribute set to None."""
        Vlog_block_base.__init__(self)
        self.tijd = None

    def process_data(self, message):
        """
        Parses the message and processes the timestamp.

        :param str message: Vlog message
        """
        try:
            if (len(message) >= 13):
                # Collect time notation
                time_notation = message[2:]
                if len(time_notation)==15: #time input writes yyyy-mm-d instead of yyyy-mm-dd so we have to change that.
                    time_notation = time_notation[:6]+'0'+time_notation[6:]

                #EDIT TOM: time in vlog data has 24:00:00 instead of 00:00:00; need to change that:
                if time_notation[8:12]=='2400':
                    time_notation = time_notation[:8]+time_notation[8:].replace("24","00")

                    #now add one day.
                    time_notation = datetime.datetime.strptime(time_notation,"%Y%m%d%H%M%S%f")
                    time_notation = time_notation +datetime.timedelta(days=1)
                    time_notation = time_notation.strftime("%Y%m%d%H%M%S%f")[:-4]

                # Process date
                year = int(time_notation[0:4])
                month = int(time_notation[4:6])
                day = int(time_notation[6:8])

                # Process time
                hour = int(time_notation[8:10])
                minute = int(time_notation[10:12])
                second = int(time_notation[12:14])
                microsecond = int(time_notation[14]) * 100000

                # Store values
                self.tijd = datetime.datetime(year, month, day, hour, minute, second, microsecond)

            self.data_processed = True
            # print(self.tijd)
        except Exception as e:
            print(e)
            print('problem with time: '+time_notation)