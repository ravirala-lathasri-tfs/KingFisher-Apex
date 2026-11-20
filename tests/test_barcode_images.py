"""Run OpenCV detection -> EfficientNet quality -> ZXing decoding."""
import argparse,csv,time
from pathlib import Path
import cv2,zxingcpp
from barcode_quality_project.src.opencv_detector import detect_barcode
from barcode_quality_project.src.opencv_quality_pipeline import crop_barcode
from barcode_quality_project.src.quality_inference import load_quality_model,predict_quality
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('images',nargs='+'); ap.add_argument('--model',default='barcode_quality_project/models/quality_best.pth'); a=ap.parse_args(); out=Path('outputs/google_pipeline_test'); out.mkdir(parents=True,exist_ok=True); model=load_quality_model(a.model); rows=[]
 for name in a.images:
  p=Path(name); im=cv2.imread(str(p)); row={'filename':str(p),'quality_prediction':'UNKNOWN','good_probability':'','bad_probability':'','detection_status':'NOT_DETECTED','decoder_status':'NOT_ATTEMPTED','decoded_value':'','barcode_format':'','final_action':'RETAKE'}
  det=detect_barcode(im) if im is not None else {'detected':False}
  if det['detected']:
   row['detection_status']='DETECTED'; crop=crop_barcode(im,det['bbox']); q=predict_quality(model,crop); row.update({'quality_prediction':q['quality'],'good_probability':q['good_probability'],'bad_probability':q['bad_probability']})
   if q['quality']=='GOOD':
    t=time.perf_counter(); d=zxingcpp.read_barcodes(crop); _=(time.perf_counter()-t)*1000; row['decoder_status']='DECODED' if d else 'NOT_DECODED'; row['final_action']='CONTINUE' if d else 'RETAKE';
    if d: row.update({'decoded_value':d[0].text,'barcode_format':str(d[0].format)})
   else: row['final_action']='RETAKE'
  print(f"\nFilename: {row['filename']}\nQuality prediction: {row['quality_prediction']}\nGOOD probability: {row['good_probability']}\nBAD probability: {row['bad_probability']}\nDetection: {row['detection_status']}\nDecoder: {row['decoder_status']}\nDecoded value: {row['decoded_value'] or 'Not available'}\nBarcode format: {row['barcode_format'] or 'Unknown'}\nFinal action: {row['final_action']}")
  rows.append(row)
 fields=['filename','quality_prediction','good_probability','bad_probability','detection_status','decoder_status','decoded_value','barcode_format','final_action']
 with (out/'results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 print('\nSummary:'); print('Total images:',len(rows)); print('Quality GOOD:',sum(r['quality_prediction']=='GOOD' for r in rows)); print('Quality BAD:',sum(r['quality_prediction']=='BAD' for r in rows)); print('Detection success:',sum(r['detection_status']=='DETECTED' for r in rows)); print('Detection failure:',sum(r['detection_status']=='NOT_DETECTED' for r in rows)); print('Decoder success:',sum(r['decoder_status']=='DECODED' for r in rows)); print('Decoder failure:',sum(r['decoder_status']=='NOT_DECODED' for r in rows)); print('Final CONTINUE:',sum(r['final_action']=='CONTINUE' for r in rows)); print('Final RETAKE:',sum(r['final_action']=='RETAKE' for r in rows)); print('GOOD + DECODED:',sum(r['quality_prediction']=='GOOD' and r['decoder_status']=='DECODED' for r in rows)); print('GOOD + NOT_DECODED:',sum(r['quality_prediction']=='GOOD' and r['decoder_status']=='NOT_DECODED' for r in rows)); print('BAD + NOT_ATTEMPTED:',sum(r['quality_prediction']=='BAD' and r['decoder_status']=='NOT_ATTEMPTED' for r in rows))
if __name__=='__main__':main()
