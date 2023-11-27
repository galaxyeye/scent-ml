import os
import sys

from pyspark.sql import SparkSession
from pyspark.mllib.feature import StandardScaler
from ml.dom.data.DataUtils import DataUtils
from pyspark.mllib.clustering import KMeans

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("kmeans").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("INFO")

path = "../../../data/dom/amazon.dataset.libsvm.11.24.50.txt"
# path = "amazon.dataset.csv.11.25.100.csv"
lines = sc.textFile(path)

columnNames = "1:top-g0 2:top-g1 3:top-g2 4:top-g3 5:left-g0 6:left-g1 7:left-g2 8:left-g3 9:width-g0 10:width-g1 " \
              "11:width-g2 12:width-g3 13:height-g0 14:height-g1 15:height-g2 16:height-g3 17:char-g0 18:char-g1 " \
              "19:char-g2 20:char-g3 21:txt_nd-g0 22:txt_nd-g1 23:txt_nd-g2 24:txt_nd-g3 25:img-g0 26:img-g1 " \
              "27:img-g2 28:img-g3 29:a-g0 30:a-g1 31:a-g2 32:a-g3 33:sibling-g0 34:sibling-g1 35:sibling-g2 " \
              "36:sibling-g3 37:child-g0 38:child-g1 39:child-g2 40:child-g3 41:dep-g0 42:dep-g1 43:dep-g2 44:dep-g3 " \
              "45:seq-g0 46:seq-g1 47:seq-g2 48:seq-g3 49:txt_dns-g0 50:txt_dns-g1 51:txt_dns-g2 52:txt_dns-g3 " \
              "53:pid-g0 54:pid-g1 55:pid-g2 56:pid-g3 57:tag-g0 58:tag-g1 59:tag-g2 60:tag-g3 61:nd_id-g0 " \
              "62:nd_id-g1 63:nd_id-g2 64:nd_id-g3 65:nd_cs-g0 66:nd_cs-g1 67:nd_cs-g2 68:nd_cs-g3 69:ft_sz-g0 " \
              "70:ft_sz-g1 71:ft_sz-g2 72:ft_sz-g3 73:color-g0 74:color-g1 75:color-g2 76:color-g3 77:b_bolor-g0 " \
              "78:b_bolor-g1 79:b_bolor-g2 80:b_bolor-g3 81:rtop-g0 82:rtop-g1 83:rtop-g2 84:rtop-g3 85:rleft-g0 " \
              "86:rleft-g1 87:rleft-g2 88:rleft-g3 89:rrow-g0 90:rrow-g1 91:rrow-g2 92:rrow-g3 93:rcol-g0 94:rcol-g1 " \
              "95:rcol-g2 96:rcol-g3 97:dist-g0 98:dist-g1 99:dist-g2 100:dist-g3 101:simg-g0 102:simg-g1 103:simg-g2 " \
              "104:simg-g3 105:mimg-g0 106:mimg-g1 107:mimg-g2 108:mimg-g3 109:limg-g0 110:limg-g1 111:limg-g2 " \
              "112:limg-g3 113:aimg-g0 114:aimg-g1 115:aimg-g2 116:aimg-g3 117:saimg-g0 118:saimg-g1 119:saimg-g2 " \
              "120:saimg-g3 121:maimg-g0 122:maimg-g1 123:maimg-g2 124:maimg-g3 125:laimg-g0 126:laimg-g1 " \
              "127:laimg-g2 128:laimg-g3 129:char_max-g0 130:char_max-g1 131:char_max-g2 132:char_max-g3 " \
              "133:char_ave-g0 134:char_ave-g1 135:char_ave-g2 136:char_ave-g3 137:own_char-g0 138:own_char-g1 " \
              "139:own_char-g2 140:own_char-g3 141:own_txt_nd-g0 142:own_txt_nd-g1 143:own_txt_nd-g2 " \
              "144:own_txt_nd-g3 145:grant_child-g0 146:grant_child-g1 147:grant_child-g2 148:grant_child-g3 " \
              "149:descend-g0 150:descend-g1 151:descend-g2 152:descend-g3 153:sep-g0 154:sep-g1 155:sep-g2 " \
              "156:sep-g3 157:rseq-g0 158:rseq-g1 159:rseq-g2 160:rseq-g3 161:txt_nd_c-g0 162:txt_nd_c-g1 " \
              "163:txt_nd_c-g2 164:txt_nd_c-g3 165:vcc-g0 166:vcc-g1 167:vcc-g2 168:vcc-g3 169:vcv-g0 170:vcv-g1 " \
              "171:vcv-g2 172:vcv-g3 173:avcc-g0 174:avcc-g1 175:avcc-g2 176:avcc-g3 177:avcv-g0 178:avcv-g1 " \
              "179:avcv-g2 180:avcv-g3 181:hcc-g0 182:hcc-g1 183:hcc-g2 184:hcc-g3 185:hcv-g0 186:hcv-g1 187:hcv-g2 " \
              "188:hcv-g3 189:ahcc-g0 190:ahcc-g1 191:ahcc-g2 192:ahcc-g3 193:ahcv-g0 194:ahcv-g1 195:ahcv-g2 " \
              "196:ahcv-g3 197:txt_df-g0 198:txt_df-g1 199:txt_df-g2 200:txt_df-g3 201:cap_df-g0 202:cap_df-g1 " \
              "203:cap_df-g2 204:cap_df-g3 205:tn_max_w-g0 206:tn_max_w-g1 207:tn_max_w-g2 208:tn_max_w-g3 " \
              "209:tn_ave_w-g0 210:tn_ave_w-g1 211:tn_ave_w-g2 212:tn_ave_w-g3 213:tn_max_h-g0 214:tn_max_h-g1 " \
              "215:tn_max_h-g2 216:tn_max_h-g3 217:tn_ave_h-g0 218:tn_ave_h-g1 219:tn_ave_h-g2 220:tn_ave_h-g3 " \
              "221:a_max_w-g0 222:a_max_w-g1 223:a_max_w-g2 224:a_max_w-g3 225:a_ave_w-g0 226:a_ave_w-g1 " \
              "227:a_ave_w-g2 228:a_ave_w-g3 229:a_max_h-g0 230:a_max_h-g1 231:a_max_h-g2 232:a_max_h-g3 " \
              "233:a_ave_h-g0 234:a_ave_h-g1 235:a_ave_h-g2 236:a_ave_h-g3 237:img_max_w-g0 238:img_max_w-g1 " \
              "239:img_max_w-g2 240:img_max_w-g3 241:img_ave_w-g0 242:img_ave_w-g1 243:img_ave_w-g2 244:img_ave_w-g3 " \
              "245:img_max_h-g0 246:img_max_h-g1 247:img_max_h-g2 248:img_max_h-g3 249:img_ave_h-g0 250:img_ave_h-g1 " \
              "251:img_ave_h-g2 252:img_ave_h-g3 253:tn_total_w-g0 254:tn_total_w-g1 255:tn_total_w-g2 " \
              "256:tn_total_w-g3 257:tn_total_h-g0 258:tn_total_h-g1 259:tn_total_h-g2 260:tn_total_h-g3 " \
              "261:a_total_w-g0 262:a_total_w-g1 263:a_total_w-g2 264:a_total_w-g3 265:a_total_h-g0 266:a_total_h-g1 " \
              "267:a_total_h-g2 268:a_total_h-g3 269:img_total_w-g0 270:img_total_w-g1 271:img_total_w-g2 " \
              "272:img_total_w-g3 273:img_total_h-g0 274:img_total_h-g1 275:img_total_h-g2 276:img_total_h-g3"
columnNames = columnNames.split(" ")
columnNames = list(map(lambda item: (item.split(":")[1]), columnNames))

print(columnNames)

metadata = lines.filter(lambda line: line[0] == "#")

rdd = lines \
    .filter(lambda line: line[0] != "#") \
    .map(lambda line: (line.split(" ^|^ "))) \
    .filter(lambda record: len(record) == 4)

kNodes = rdd.filter(lambda r: "B0073UBRP2" in r[3]).count()

print("kNodes " + str(kNodes))

kNodes2 = rdd.toDF(["label", "numeric_features", "text", "url"]).groupBy("url").count()

rdd2 = rdd.map(lambda r: DataUtils.parse_libsvm_line_to_label_vector(276, r[0] + " " + r[1]))

df2 = rdd2.toDF(["label", "features"])
df2.printSchema()

label = rdd2.map(lambda x: x[0])
features = rdd2.map(lambda x: x[1])
featureDF = features.toDF(columnNames)

scaler = StandardScaler()
scalerModel = scaler.fit(features)
scaledFeatureRDD = scalerModel.transform(features)

kmeans = KMeans()
kmeansModel = kmeans.train(scaledFeatureRDD, kNodes)

print("Final centers: " + str(kmeansModel.clusterCenters))
print("Total Cost: " + str(kmeansModel.computeCost(scaledFeatureRDD)))

spark.stop()
