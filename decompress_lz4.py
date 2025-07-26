import os
import struct
import lz4.frame
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def decompress_large_file(input_file, output_file=None):
    try:
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            return False
            
        if output_file is None:
            output_file = input_file[:-4]  # 移除 .lz4 后缀
            
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(input_file, 'rb') as f_in:
            header = f_in.read(8)
            original_size = struct.unpack('!Q', header)[0]

            with open(output_file, 'wb') as f_out:
                while True:
                    size_header = f_in.read(4)
                    if not size_header:
                        break
                    
                    chunk_size = struct.unpack('!I', size_header)[0]
                    compressed_chunk = f_in.read(chunk_size)
                    if not compressed_chunk:
                        break
                    
                    decompressed = lz4.frame.decompress(compressed_chunk)
                    f_out.write(decompressed)

        decompressed_size = os.path.getsize(output_file)
        if decompressed_size != original_size:
            logger.error(f"解压后文件大小不匹配: 原始={original_size}, 解压后={decompressed_size}")
            return False

        logger.info(f"文件解压完成: {output_file}")
        return True

    except Exception as e:
        logger.error(f"解压文件失败: {str(e)}")
        return False

# Example usage
decompress_large_file("/Users/shuzongfei/Downloads/2025-06-01T00_00_00_cn_flatted.nc.lz4")
