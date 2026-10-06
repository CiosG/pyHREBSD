"""Lazy reader for Thermo Fisher xTalView ``.tfs.hdf5`` EBSD exports."""
from __future__ import annotations
import math
from pathlib import Path
import h5py
import numpy as np
from .geometry import euler_to_matrix, phosphor_to_sample_from_oxford

def _bunge_matrix(e):
    p1,p,p2=e; c1,s1,c,s,c2,s2=math.cos(p1),math.sin(p1),math.cos(p),math.sin(p),math.cos(p2),math.sin(p2)
    return np.array(((c1*c2-s1*s2*c,-c1*s2-s1*c2*c,s1*s),(s1*c2+c1*s2*c,-s1*s2+c1*c2*c,-c1*s),(s2*s,c2*s,c)))
def _rotation_x(a):
    c,s=math.cos(a),math.sin(a); return np.array(((1.,0.,0.),(0.,c,-s),(0.,s,c)))
def _matrix_to_bunge(m):
    p=math.acos(float(np.clip(m[2,2],-1.,1.)))
    if abs(math.sin(p))>1e-10: p1,p2=math.atan2(m[0,2],-m[1,2]),math.atan2(m[2,0],m[2,1])
    else: p1,p2=math.atan2(m[1,0],m[0,0]),0.
    return np.mod((p1,p,p2),2.*math.pi)

class TFSReader:
    """Read xTalView EBSD maps lazily with the common PyHREBSD API."""
    source_format="tfs"
    def __init__(self,path: str|Path,pattern_type="processed",pc_source="ebsd"):
        self.path=Path(path)
        if pattern_type not in ("processed","auto"): raise ValueError("TFS HDF5 contains processed patterns only")
        if pc_source!="ebsd": raise ValueError("TFS reader uses the EBSD PatternCenter dataset")
        self.pattern_type,self.pc_source="processed","ebsd"; self.pc_source_path="/Site/EBSD/MapData/PatternCenter"; self.scan_group=None
        self._file=h5py.File(self.path,"r")
        try:
            patterns=self._file.get("Site/EBSD/Patterns/Processed")
            if patterns is None or patterns.ndim!=4: raise KeyError("missing /Site/EBSD/Patterns/Processed (rows, columns, height, width)")
            self._patterns=patterns; self.y_cells,self.x_cells,self.height,self.width=map(int,patterns.shape); self.count=self.x_cells*self.y_cells
            data=self._file.get("Site/EBSD/MapData")
            if data is None: raise KeyError("missing /Site/EBSD/MapData")
            self._map_data=data
            for name,shape in (("PatternCenter",(self.y_cells,self.x_cells,3)),("EulerAngles",(self.y_cells,self.x_cells,3)),("Phase",(self.y_cells,self.x_cells))):
                if name not in data or tuple(data[name].shape)!=shape: raise ValueError(f"/Site/EBSD/MapData/{name} has unexpected shape")
            acquisition=self._file.get("Site/Acquisition"); self._specimen_tilt=self._scalar(acquisition,"SpecimenTilt"); step=self._scalar(acquisition,"StepSize")
            if step is None or step<=0: raise ValueError("TFS StepSize is missing or invalid")
            self._header={"X Step":np.asarray([step*1e6]),"Y Step":np.asarray([step*1e6])}
            dcs=self._vector(self._file.get("Site/EBSD/Info"),"DCStoSCS",3)
            self._detector_euler=None if dcs is None or self._specimen_tilt is None else _matrix_to_bunge(_rotation_x(self._specimen_tilt)@_bunge_matrix(np.deg2rad(np.mod(dcs,360.))))
        except Exception:
            self._file.close(); raise
    @staticmethod
    def _scalar(group,name):
        if group is None or name not in group:return None
        a=np.asarray(group[name][()]).reshape(-1); return float(a[0]) if len(a) else None
    @staticmethod
    def _vector(group,name,size):
        if group is None or name not in group:return None
        a=np.asarray(group[name][()],dtype=float).reshape(-1); return a if a.size==size and np.all(np.isfinite(a)) else None
    def _check_index(self,index):
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)):raise TypeError("pattern index must be an integer")
        if index<0 or index>=self.count:raise IndexError(f"pattern index {index} outside 0..{self.count-1}")
    def available_indices(self):return np.arange(self.count,dtype=np.int64)
    def has_pattern(self,index):return not isinstance(index,(bool,np.bool_)) and isinstance(index,(int,np.integer)) and 0<=index<self.count
    def uncropped_pattern(self,index):
        self._check_index(index); row,col=divmod(index,self.x_cells); return np.asarray(self._patterns[row,col],dtype=np.float64)
    def pattern(self,index):
        a=self.uncropped_pattern(index); side=min(self.height,self.width); top,left=(self.height-side)//2,(self.width-side)//2; return a[top:top+side,left:left+side]
    def pattern_center(self,index):
        self._check_index(index); row,col=divmod(index,self.x_cells); x,y,z=np.asarray(self._map_data["PatternCenter"][row,col],dtype=float)
        if not np.all(np.isfinite((x,y,z))):return None
        side=min(self.height,self.width); left=(self.width-side)//2; bottom=self.height-side-(self.height-side)//2
        return ((x*self.width-left)/side,(y*self.height-bottom)/side,z*self.height/side)
    def pattern_centers(self):
        c=np.asarray(self._map_data["PatternCenter"][:],dtype=float).reshape(self.count,3); r=c.copy(); side=min(self.height,self.width); left=(self.width-side)//2; bottom=self.height-side-(self.height-side)//2
        r[:,0]=(c[:,0]*self.width-left)/side; r[:,1]=(c[:,1]*self.height-bottom)/side; r[:,2]=c[:,2]*self.height/side; r[~np.isfinite(r).all(axis=1)]=np.nan; return r
    def orientation(self,index):
        self._check_index(index); row,col=divmod(index,self.x_cells); a=np.asarray(self._map_data["EulerAngles"][row,col],dtype=float)
        if not np.all(np.isfinite(a)):raise ValueError(f"invalid Euler angles at pattern index {index}")
        return euler_to_matrix(*np.deg2rad(a))
    def grain_inputs(self):
        e=np.deg2rad(np.asarray(self._map_data["EulerAngles"][:],dtype=float)).reshape(self.count,3); p=np.asarray(self._map_data["Phase"][:],dtype=np.int32).reshape(self.count); q=np.asarray(self._map_data["IndexQuality"][:],dtype=float).reshape(self.count) if "IndexQuality" in self._map_data else None; return e,p,q
    def map_index(self,column,row):
        if column<0 or row<0 or column>=self.x_cells or row>=self.y_cells:raise IndexError("map coordinate outside TFS scan")
        return int(row*self.x_cells+column)
    def sample_tilt_degrees(self):return None if self._specimen_tilt is None else math.degrees(self._specimen_tilt)
    def camera_elevation_degrees(self):return None if self._detector_euler is None else math.degrees(self._detector_euler[1])-90.
    def phosphor_to_sample(self,sample_tilt_degrees=None):
        tilt=self.sample_tilt_degrees() if sample_tilt_degrees is None else float(sample_tilt_degrees)
        return None if tilt is None or self._detector_euler is None else phosphor_to_sample_from_oxford(math.radians(tilt),self._detector_euler)
    def has_unprocessed_static_background(self):return False
    def unprocessed_static_background(self):raise KeyError("TFS HDF5 contains no unprocessed static background")
    def close(self):self._file.close()
    def __enter__(self):return self
    def __exit__(self,exc_type,exc_value,traceback):self.close()
