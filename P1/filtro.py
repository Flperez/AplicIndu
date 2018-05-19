import numpy as np

class filtro:

    #Umbral de solapamiento
    overlapThresh = 0.8

    def __init__(self,boxes):
        self.__boxes = boxes


    def non_max_suppression_fast(self,boxes):
        '''
            Código copiado de:
            https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python
            Se utilizara para refinar los rectangulos

            :param boxes:
            :param overlapThresh:
            :return:
            '''
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > self.overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    def unique_rows(self,a):
        '''
        Eliminar las filas duplicadas
        https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
        '''
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

    def contained(self,A, B):
        if (A[0] >= B[0] and A[1] >= B[1]
            and A[2] <= B[2] and A[3] <= B[3]):
            return True
        else:
            return False

    def delete_rects_contained(self,rects):
        '''
        Eliminamos las BB contenidas dentro de otras BB
        :param rects:
        :return:
        '''
        # Vector que contiene los indices que contienen las filas que deben eliminarse
        deleted = np.array([])

        # Calculamos las areas ya que una BB contenida en otra tendra una area menor
        areas = [(rects[k, 0] - rects[k, 2]) * (rects[k, 1] - rects[k, 3]) for k in range(0, len(rects))]

        # Eliminamos los rectangulos contenidos dentro de otros rectangulos
        for i in range(0, len(rects)):
            for j in range(0, len(rects)):
                if areas[i] <= areas[j] and i != j:
                    if filtro.contained(self,rects[i], rects[j]) == True:
                        deleted = np.append(deleted, i)

        deleted = deleted.astype(int)
        rects = np.delete(rects, (deleted), axis=0)
        return rects

    def filtrar(self):
        #1º Eliminamos los rectangulos duplicados
        rects_no_duplicados = filtro.unique_rows(self=self,a=self.__boxes)
        #2º Refinamos los rectangulos
        rects_non_maxima = filtro.non_max_suppression_fast(self=self,boxes=rects_no_duplicados)
        #3º Eliminamos los contenidos
        rects_filtrados = filtro.delete_rects_contained(self=self,rects=rects_non_maxima)
        return rects_filtrados





























