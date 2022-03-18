import re
import psutil
import PyNvCodec as nvc
from enum import Enum
import numpy as np
import time
import utils
from utils import IterationResult, plot_list_to_image
import gpustat
import csv

PROCESS_NAME = "videc-benchmark"
PROCESS_PID = 0


class DecodeStatus(Enum):
    # Decoding error.
    DEC_ERR = 0,
    # Frame was submitted to decoder.
    # No frames are ready for display yet.
    DEC_SUBM = 1,
    # Frame was submitted to decoder.
    # There's a frame ready for display.
    DEC_READY = 2


class NvDecoder:
    def __init__(self, gpu_id: int, enc_file: str):
        # Initialize decoder with built-in demuxer.
        self.nv_dmx = None
        self.nv_dec = nvc.PyNvDecoder(enc_file, gpu_id)

        self.gpu_id = gpu_id
        # Frame to seek to next time decoding function is called.
        # Negative values means 'don't use seek'.  Non-negative values mean
        # seek frame number.
        self.sk_frm = int(-1)
        # Total amount of decoded frames
        self.num_frames_decoded = int(0)
        # Numpy array to store decoded frames pixels
        self.frame_nv12 = np.ndarray(shape=(0), dtype=np.uint8)
        # Encoded packet data
        self.packet_data = nvc.PacketData()
        # Seek mode
        self.seek_mode = nvc.SeekMode.PREV_KEY_FRAME

    # Returns number of decoded frames.
    def dec_frames(self) -> int:
        return self.num_frames_decoded

    # Returns number of frames in video.
    def stream_num_frames(self) -> int:
        return self.nv_dec.Numframes()

    def decode_frame_builtin(self, verbose=False) -> DecodeStatus:
        status = DecodeStatus.DEC_ERR

        try:
            frame_ready = False
            frame_cnt_inc = 0

            if self.sk_frm >= 0:
                print('Seeking for the frame ', str(self.sk_frm))
                seek_ctx = nvc.SeekContext(int(self.sk_frm), self.seek_mode,
                                           self.seek_criteria)
                self.sk_frm = -1

                frame_ready = self.nv_dec.DecodeSingleFrame(self.frame_nv12,
                                                            seek_ctx, self.packet_data)
                frame_cnt_inc = seek_ctx.num_frames_decoded
            else:
                frame_ready = self.nv_dec.DecodeSingleFrame(
                    self.frame_nv12, self.packet_data)
                frame_cnt_inc = 1

            # Nvdec is sync in this mode so if frame isn't returned it means
            # EOF or error.
            if frame_ready:
                self.num_frames_decoded += 1
                status = DecodeStatus.DEC_READY

                if verbose:
                    print('Decoded ', frame_cnt_inc, ' frames internally')
            else:
                return status

            if verbose:
                print("frame pts (display order)      :", self.packet_data.pts)
                print("frame dts (display order)      :", self.packet_data.dts)
                print("frame pos (display order)      :", self.packet_data.pos)
                print("frame duration (display order) :",
                      self.packet_data.duration)
                print("")

        except Exception as e:
            print(getattr(e, 'message', str(e)))

        return status

    # Decode single video frame
    def decode_frame(self, verbose=False) -> DecodeStatus:
        return self.decode_frame_builtin(verbose)

    # Decode all available video frames and write them to output file.
    def decode(self, frames_to_decode=-1, verbose=False, dump_frames=True, with_plot=False) -> IterationResult:
        frame_decode_record = []
        cpu_util_record = []
        mem_util_record = []
        gpu_util_record = []
        gpu_mem_util_record = []

        process_name = re.compile(PROCESS_NAME)
        for p in psutil.process_iter(['pid', 'name', 'memory_info']):
            if not process_name.match(p.name()):
                continue
            PROCESS_PID = p.pid
        # gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu_id)
        psutil_handle = psutil.Process(PROCESS_PID)

        frame_count = 0
        # Main decoding cycle
        while (self.dec_frames() < frames_to_decode) if (frames_to_decode > 0) else True:
            gpu = gpustat.core.GPUStatCollection.new_query()
            gpu_processes = filter(lambda x: process_name.match(
                x['command']), gpu[0].processes)

            start_counter = time.perf_counter()
            status = self.decode_frame(verbose)
            if status == DecodeStatus.DEC_ERR:
                break
            elif dump_frames and status == DecodeStatus.DEC_READY:
                frame_count += 1
            end_counter = time.perf_counter()

            processing_time = round(utils.s_to_ms(
                end_counter - start_counter), 2)
            cpu_util = psutil.cpu_percent()
            mem_util = round(utils.b_to_mb(psutil_handle.memory_info().rss), 2)
            gpu_util = gpu[0].utilization

            for process in gpu_processes:
                gpu_mem_util_record.append(process["gpu_memory_usage"])

            frame_decode_record.append(processing_time)
            cpu_util_record.append(cpu_util)
            mem_util_record.append(mem_util)
            gpu_util_record.append(gpu_util)

            # Dump records to CSV files
            with open("benchmark-results/csv/fpt/nvdec.csv", 'w') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow(frame_decode_record)
            with open("benchmark-results/csv/cpu/nvdec.csv", 'w') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow(cpu_util_record)
            with open("benchmark-results/csv/mem/nvdec.csv", 'w') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow(mem_util_record)
            with open("benchmark-results/csv/gpu/nvdec.csv", 'w') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow(gpu_util_record)
            with open("benchmark-results/csv/gpu-mem/nvdec.csv", 'w') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow(gpu_mem_util_record)

        if with_plot:
            plot_list_to_image(
                frame_decode_record, 'benchmark-results/plot/fpt/nvdec.png')
            plot_list_to_image(
                cpu_util_record, 'benchmark-results/plot/cpu/nvdec.png')
            plot_list_to_image(
                mem_util_record, 'benchmark-results/plot/mem/nvdec.png')
            plot_list_to_image(
                gpu_util_record, 'benchmark-results/plot/gpu/nvdec.png')
            plot_list_to_image(
                gpu_mem_util_record, 'benchmark-results/plot/gpu-mem/nvdec.png')

        return {
            "processing_time": frame_decode_record,
            "cpu": cpu_util_record,
            "mem": mem_util_record,
            "gpu": gpu_util_record,
            "gpu_mem": gpu_mem_util_record
        }
