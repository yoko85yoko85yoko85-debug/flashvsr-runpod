import json
from pathlib import Path

def build():
    note='''## FlashVSR+ Turbo：アップスケール専用

1. Load Videoで動画をアップロード。
2. 倍率を1.5倍・2倍・4倍から選択。
3. Queueを実行すると、拡大後のMP4を保存・プレビューします。

初期値：2倍・標準（速度優先）・シード1。
公式モデル推奨は4倍です。1.5倍・2倍は実用サイズで比較する選択肢です。

30秒まで・固定fps1〜60・SDR動画。出力各辺2048pxまで。
例：1280×720は1.5倍で1920×1080。800×800は2倍で1600×1600。
小数倍率の出力は、保存形式に合わせて2px刻みに丸めます。
入力512MiBまで。アプリ版の64MiB転送制限はありません。
縦横比・フレーム数・fpsを保ち、音声をAACで引き継ぎます。
細部が変わる可能性があるため、元映像と比較してください。

ComfyUIのoutput/FlashVSR-PodにMP4と処理情報JSONを保存します。
時間・解像度によりGPUメモリ不足や時間切れになる場合があります。
初回はモデル読込とTritonのコンパイルで時間が増えます。
'''
    nodes=[
      {'id':1,'type':'LoadVideo','title':'元動画を読み込む','pos':[0,0],'size':[420,360],
       'flags':{},'order':0,'mode':0,'inputs':[],
       'outputs':[{'name':'VIDEO','type':'VIDEO','links':[1],'slot_index':0}],
       'properties':{'Node name for S&R':'LoadVideo'},'widgets_values':['flashvsr-example.mp4','video']},
      {'id':2,'type':'FlashVSRPodTurbo','pos':[500,0],'size':[480,520],
       'flags':{},'order':1,'mode':0,'inputs':[{'name':'video','type':'VIDEO','link':1}],
       'outputs':[{'name':'VIDEO','type':'VIDEO','links':[],'slot_index':0}],
       'properties':{'Node name for S&R':'FlashVSRPodTurbo'},
       'widgets_values':['2倍','標準（速度優先）',1,'fixed']},
      {'id':3,'type':'MarkdownNote','title':'使い方','pos':[0,420],'size':[460,670],
       'flags':{},'order':2,'mode':0,'inputs':[],'outputs':[],'properties':{},'widgets_values':[note]}]
    return {'last_node_id':3,'last_link_id':1,'nodes':nodes,'links':[[1,1,0,2,0,'VIDEO']],
            'groups':[],'config':{},'extra':{'ds':{'scale':.8,'offset':[60,60]}},'version':.4}

def write(comfy):
    dest=Path(comfy)/'user/default/workflows';dest.mkdir(parents=True,exist_ok=True)
    (dest/'FlashVSR-Pod-Turbo.json').write_text(json.dumps(build(),ensure_ascii=False,indent=2))

if __name__=='__main__':
    import sys
    write(sys.argv[1])
