import psutil
import PyNvCodec as nvc
from enum import Enum
import numpy as np
import time
import utils
from utils import IterationResult
import gpustat


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
    def decode(self, frames_to_decode=-1, verbose=False, dump_frames=True, current_iteration=1) -> IterationResult:
        frame_decode_record = []
        cpu_util_record = []
        mem_util_record = []
        gpu_util_record = []
        gpu_mem_util_record = []

        # gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu_id)

        frame_count = 0
        # Main decoding cycle
        while (self.dec_frames() < frames_to_decode) if (frames_to_decode > 0) else True:
            gpu = gpustat.core.GPUStatCollection.new_query()
            start_counter = time.perf_counter()
            status = self.decode_frame(verbose)
            if status == DecodeStatus.DEC_ERR:
                break
            elif dump_frames and status == DecodeStatus.DEC_READY:
                frame_count += 1
            end_counter = time.perf_counter()

            processing_time = utils.s_to_ms(end_counter - start_counter)
            cpu_util = psutil.cpu_percent()
            mem_util = utils.b_to_mb(psutil.virtual_memory().used)
            # gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(gpu_handle)
            gpu_util = gpu[0].utilization
            gpu_mem_util = gpu[0].memory_used

            frame_decode_record.append(processing_time)
            cpu_util_record.append(cpu_util)
            mem_util_record.append(mem_util)
            gpu_util_record.append(gpu_util)
            gpu_mem_util_record.append(gpu_mem_util)

        utils.plot_list_to_image(
            cpu_util_record, 'benchmark-results/plot/cpu/nvcuvid-cpu-{}.png'.format(current_iteration))

        return {
            "processing_time": frame_decode_record,
            "cpu": cpu_util_record,
            "mem": mem_util_record,
            "gpu": gpu_util_record,
            "gpu_mem": gpu_mem_util_record
        }
