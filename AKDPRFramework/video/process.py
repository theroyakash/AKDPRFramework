import av

class Extractor(object):
    def __init__(self):
        pass

    def extract_frames(self, path):
        '''
        Extract frames from a video from a given path.
            
            Args:
                - path: path to the video
        
            Usage:
                - Import like this: ``from AKDPRFramework.video import process as p``
                - Then use the exractor class like: ``extractor = p.Extractor()``
                - Now mention the path in the ``extract_path`` method like: ``extractor.extract_frames(path='/path')``
        '''
        video = av.open(path)
        for frame in video.decode(0):
            yield frame.to_image()