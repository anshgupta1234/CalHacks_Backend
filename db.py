import singlestoredb as s2

#connection instructions
conn_link = 'https://admin:Calhacks123@host:port/database?local_infile=True'
host = 'svc-237c8144-15b8-45fa-b7f9-0c2f105c6881-dml.aws-virginia-6.svc.singlestore.com'
port = '3306'
user = 'admin'
password = 'Calhacks123'

conn = s2.connect(host=host, port=port, user=user, password=password, database='openaidb', results_type='tuple')

'''with conn:
    with conn.cursor() as cur:
        cur.execute('INSERT INTO myvectortable (text, vector) VALUES ("The rules should work for them.", JSON_ARRAY_PACK("[0.00053209817269817, -0.00770887266844511, 0.002101344056427479, -0.028238818049430847, -0.005288016051054001, 0.018149660900235176, -0.0016474333824589849, -0.021246735006570816, -0.005467213690280914, -0.038409117609262466, 0.03248545899987221, -0.014809150248765945, 0.022707363590598106, -0.02373521216213703, -0.03010517545044422, 0.012117806822061539, 0.02503354847431183, -0.021314356476068497, 0.028292914852499962, -0.006424060557037592, -0.005964233074337244, 0.011184627190232277, -0.008811105974018574, -0.022585643455386162, -0.029023228213191032, -0.02234220691025257, 0.018595963716506958, -0.02687285840511322, 0.007269331719726324, -0.021003296598792076, 0.0281576719135046, 0.013118607923388481, -0.018230807036161423, 0.001089554512873292, -0.020381176844239235, -0.011292821727693081, 0.006096095312386751, -0.021841805428266525, 0.013794824481010437, 0.009338554926216602, 0.03156580403447151, -0.008993684314191341, 0.004337931517511606, -0.05012119561433792, -0.016242729499936104, 0.021246735006570816, 0.0020337223540991545, -0.019191036000847816, -0.025993777438998222, -0.0029432340525090694, 0.031024830415844917, 0.008750246837735176, -0.038652557879686356, -0.005791797768324614, -0.005274491850286722, 0.0005506941233761609, -0.02446552738547325, 0.033783797174692154, 0.0038341498002409935, 0.014660382643342018, 0.005923660006374121, 0.006829790771007538, 0.0048383320681750774, 0.0069582718424499035, -0.004598274827003479, -0.016689032316207886, 0.014457517303526402, 0.0009035948314704001, -0.008188986219465733, 0.022112293168902397, 0.004530653357505798, -0.009514371864497662, 0.01928570494055748, 0.004165496211498976, 0.010488123632967472, 0.005152772646397352, -0.002628793241456151, -0.027508502826094627, -0.007046179845929146, 0.01747344434261322, 0.007161136716604233, -0.011820271611213684, -0.028184719383716583, 0.0051933457143604755, 0.0007746909977868199, 0.005338732153177261, -0.01946152187883854, 0.029807640239596367, -0.01361224614083767, -0.014565711840987206, -0.003972774371504784, 0.008594716899096966, 0.01553946454077959, 0.006515349727123976, -0.013551386073231697, 0.0318092443048954, 0.005108818411827087, 0.03873370215296745, 0.0017750693950802088, -0.00830394309014082, 0.02088157832622528, 0.02541223168373108, -0.008351278491318226, -0.00504795927554369, -0.0029787353705614805, -0.01355814840644598, 0.006177241448312998, -0.021909426897764206, 0.025533949956297874, -0.02603434957563877, -0.010494885966181755, 0.025939680635929108, 3.502592153381556e-05, -0.039734505116939545, 0.004023490473628044, 0.012645255774259567, -0.010346118360757828, -0.033053480088710785, -0.04124923050403595, -0.0030903113074600697, 0.03267480060458183, 0.0034453249536454678, 0.016580838710069656, -0.005572027061134577, -0.003523089922964573, -0.001156330923549831, -0.010873567312955856, -0.008128127083182335, -0.0009467036579735577, 0.0001824728969950229, 0.039355821907520294, 0.011042621918022633, 0.01324708852916956, -0.0008042754488997161, -0.014038262888789177, 0.012300385162234306, 0.010264972224831581, 0.009710474871098995, -0.003222173545509577, -0.013531100004911423, 0.022950801998376846, 0.01639149710536003, 0.0036414279602468014, -0.018217282369732857, 0.012679066509008408, 0.010718037374317646, -0.01886645145714283, -0.0009086664649657905, -0.009352079592645168, -0.009723998606204987, 0.00610961951315403, -0.0035366143565624952, 0.009068068116903305, -0.007823829539120197, 0.025493375957012177, 0.00904778204858303, 0.003381084417924285, 0.014187030494213104, -0.009325031191110611, -0.012733164243400097, -0.0010650416370481253, -0.01289545651525259, 0.0043649799190461636, 0.002829967765137553, 0.031619902700185776, 0.013280900195240974, 0.017459919676184654, 0.026926957070827484, 0.006633687764406204, 0.013125370256602764, 0.017730407416820526, 0.03735421970486641, -0.03218792378902435, 0.018460720777511597, 0.008330992422997952, 0.014687431044876575, 0.0032086491119116545, 0.002241658978164196, -0.013625770807266235, 0.014349322766065598, -0.009162738919258118, 0.004652372095733881, 0.008364803157746792, 0.023829882964491844, -0.012543823570013046, -0.006440965924412012, -0.00012034547398798168, 0.003053119173273444, 0.02200409770011902, -0.004980337340384722, 0.013179467059671879, 0.01935332827270031, -0.010941189713776112, 0.007965834811329842, -0.6708071827888489, -0.054286692291498184, 0.028590450063347816, -0.007309904787689447, 0.006569446995854378, 0.0273597352206707, -0.002884065033867955, -0.009399414993822575, -0.022856131196022034, 0.027210967615246773, -0.032350216060876846, 0.010102679952979088, 0.014416944235563278, -0.03194448724389076, -0.012469439767301083, -0.03362150490283966, 0.01733820140361786, -0.0002052952186204493, -0.006870363838970661, 0.042899198830127716, -0.015093160793185234, 0.010650415904819965, -0.019623814150691032, -0.0010050273267552257, -0.015012014657258987, 0.012361245229840279, 0.000807233911473304, -0.0211926382035017, -0.013984165154397488, 0.004469793755561113, -0.02886093780398369, 0.02074633352458477, -0.01155654713511467, 0.012090758420526981, 0.03497393801808357, 0.008959873579442501, -0.023397104814648628, -0.004652372095733881, 0.014214078895747662, 0.011529497802257538, -0.022288108244538307, -0.015620609745383263, 0.026791714131832123, -0.00196610065177083, 0.0033929182682186365, 0.02067871205508709, -0.009237122721970081, 0.007424861658364534, -0.009906577877700329, 0.01848777011036873, 0.0068906499072909355, -0.009521134197711945, 0.017703358083963394, 0.01760868728160858, 0.007762969937175512, -0.0009636090835556388, 0.0085811922326684, -0.003215411212295294, 0.003212030278518796, -0.0027724893298000097, 0.009676664136350155, -0.011982562951743603, -0.032106779515743256, -0.01594519428908825, -0.011975801549851894, 0.011969039216637611, -0.01010944228619337, 0.02722449228167534, 0.010677464306354523, -0.01935332827270031, 0.01743287220597267, 0.002816443331539631, -0.01649969257414341, -0.005342113319784403, -0.013010413385927677, 0.007783256471157074, 0.013382332399487495, 0.0035805683583021164, -0.004949907772243023, 0.028184719383716583, -0.020529944449663162, -0.03191743791103363, -0.013395857065916061, 0.004892429336905479, 0.025560999289155006, 0.0035501387901604176, -0.0203947015106678, -0.010785659775137901, 0.02353234775364399, 0.003813863266259432, 0.031836289912462234, 0.010636892169713974, 0.04419753700494766, 0.0026964149437844753, 0.0203947015106678, -0.008784057572484016, -0.006241481751203537, -0.021706562489271164, 0.027724891901016235, -0.015485366806387901, 0.010711275972425938, 0.0013076344039291143, 0.016837799921631813, -0.0008587954798713326, -0.002507074037566781, 0.008074029348790646, -0.004571225959807634, 0.01743287220597267, 0.03070024773478508, -0.021557794883847237, 0.0175410658121109, -0.006505206692963839, -0.003061572089791298, -0.0055923135951161385, -0.022545071318745613, -0.032458409667015076, 0.010461075231432915, -0.007769732270389795, -0.003046357072889805, -0.030483856797218323, 0.020313555374741554, 0.01420055516064167, 0.01113729178905487, -0.026480654254555702, 0.010400216095149517, 0.013903019018471241, 0.0075533427298069, -0.010312307626008987, 0.004814664367586374, 0.003911914769560099, 0.004219593480229378, 0.02315366640686989, 0.023897504433989525, -0.0085811922326684, 0.019988970831036568, -0.013720440678298473, 0.0192180834710598, -0.014903820119798183, 0.03740831837058067, -0.01698656752705574, 0.000343391380738467, -0.007309904787689447, 0.002074295422062278, -0.010751849040389061, -0.008405376225709915, -0.018528342247009277, -0.022747935727238655, -0.008175462484359741, -0.019759057089686394, 0.010704513639211655, -0.0038882470689713955, -0.008418899960815907, -0.004094493109732866, -0.00027492441586218774, -0.022261060774326324, -0.005004005040973425, -0.003203577594831586, -0.009690187871456146, -0.012077233754098415, -0.028969131410121918, 0.001403150032274425, 0.022288108244538307, -0.0391935296356678, -0.02426266297698021, 0.019718484953045845, -0.010853281244635582, 0.0101973507553339, -0.00456784525886178, 0.017067713662981987, -0.02996993251144886, -0.007681823801249266, -0.01594519428908825, 0.01004182081669569, 0.00032627466134727, -0.008283657021820545, 0.014579236507415771, -0.005798559635877609, 0.004794377833604813, 0.01493086852133274, -0.006484920158982277, 0.0039964416064321995, -0.008554143831133842, -0.013666342943906784, -0.006140049546957016, 0.02638598345220089, -0.00022864583297632635, 0.01420055516064167, 0.019894301891326904, -0.01388949528336525, 0.01498496625572443, -0.008013170212507248, 0.01309155859053135, -0.00019789910584222525, 0.009805144742131233, 0.02134140580892563, 0.004726755898445845, 0.006383487489074469, 0.011319871060550213, -0.0122530497610569, 0.015958718955516815, 0.029320765286684036, -0.017973845824599266, 0.01296307798475027, -0.03143056109547615, -0.013233564794063568, -0.019096365198493004, 0.009250647388398647, -0.022599168121814728, 0.017392298206686974, 0.01455218717455864, 0.012888694182038307, -0.02067871205508709, -0.002180799376219511, 0.0035399955231696367, -0.0067215957678854465, 0.016242729499936104, 0.0051460107788443565, 0.024830684065818787, 0.00880434364080429, 0.016472643241286278, 0.0022467307280749083, -0.010035058483481407, 0.02700810320675373, 0.01367986761033535, 0.009142451919615269, -0.009433225728571415, -0.001254382310435176, 0.025209365412592888, 0.02178770862519741, -0.01112376805394888, -0.009446750394999981, -0.006528873927891254, 0.009277695789933205, 0.05436783656477928, 0.014903820119798183, -0.008716436102986336, 0.027657270431518555, -0.002958448836579919, 0.007729159202426672, -0.01388949528336525, 0.02074633352458477, 0.008993684314191341, 0.02105739340186119, -0.008959873579442501, 0.029050277546048164, -0.00849328376352787, 0.04522538557648659, 0.02046232298016548, 8.974401680461597e-06, 0.01455218717455864, 0.008175462484359741, 0.008074029348790646, -0.030294517055153847, 0.01915046200156212, 0.012800785712897778, -0.007343715522438288, 0.016459118574857712, 0.0033236059825867414, 0.01785212568938732, 0.03581244498491287, 0.002851944649592042, 0.013903019018471241, 0.012645255774259567, -0.01942094974219799, 0.007208472117781639, -0.005244061816483736, -0.005274491850286722, -0.017284104600548744, -0.007877926342189312, -0.014565711840987206, -0.005703889299184084, 0.005365781020373106, 0.005487500224262476, 0.0046557532623410225, 0.03467640280723572, 0.018109088763594627, -0.013274137862026691, -0.011948752216994762, 0.027265064418315887, 0.02649417705833912, -0.019623814150691032, -0.01897464506328106, 0.004689563997089863, 0.006491682026535273, -0.007783256471157074, -0.011292821727693081, -0.04925563931465149, 0.0025425755884498358, -0.012083996087312698, 0.025155268609523773, 0.007377526257187128, 0.010434026829898357, -0.002180799376219511, 0.015160782262682915, 0.010961475782096386, 0.003678619861602783, 0.006877125706523657, -0.00839185155928135, -0.020421750843524933, -0.014511614106595516, -0.004730137065052986, -0.009507609531283379, -0.0026068161241710186, -0.01694599539041519, 0.013774538412690163, 0.008520333096385002, -0.015823476016521454, -0.023234812542796135, 0.001139425439760089, -0.01535012386739254, 0.010224399156868458, -0.014971441589295864, -0.0067892177030444145, -0.002858706982806325, -0.00842566229403019, 0.007255807053297758, -0.037300124764442444, -0.0031917437445372343, 0.013044224120676517, -0.026791714131832123, -0.012516774237155914, -0.00677907420322299, -0.015282501466572285, -0.019650863483548164, 0.10164892673492432, -0.013903019018471241, 0.012510012835264206, 0.02468191646039486, -0.025114694610238075, 0.00621443334966898, -0.010204113088548183, -0.02067871205508709, 0.01935332827270031, -0.0029330907855182886, 0.019326278939843178, 0.0066674984991550446, -0.027508502826094627, -0.0140788359567523, 0.015580037608742714, 0.024316759780049324, 0.013456716202199459, -0.01802794262766838, 0.009494084864854813, -0.017202958464622498, -0.010778897441923618, 0.022572120651602745, 0.01893407292664051, 0.012056946754455566, -0.006495063193142414, -0.015593561343848705, 0.021354928612709045, 0.024965927004814148, -0.004375123418867588, -0.02607492357492447, -0.00547397555783391, 0.022856131196022034, 0.006785836536437273, 0.013788062147796154, -0.018149660900235176, 0.019623814150691032, -0.0028637784998863935, 0.002912804251536727, 0.0200160201638937, -0.021206161007285118, 0.022774985060095787, 0.007526293862611055, -8.016868378035724e-06, -0.014024738222360611, 0.018961122259497643, -0.01583699882030487, -0.01973200961947441, 0.03032156452536583, 0.007255807053297758, -0.01632387563586235, -0.0020049831364303827, -0.030132224783301353, -0.007228758651763201, 0.002101344056427479, -0.0030987639911472797, 0.005879705771803856, 0.0051865833811461926, 0.01340261846780777, 0.007343715522438288, -0.004794377833604813, -0.01740582287311554, -0.004331169184297323, 0.0023718306329101324, 0.007221996318548918, 0.007228758651763201, -0.0273597352206707, -0.029699446633458138, -0.01541774533689022, -0.02461429499089718, -0.020286506041884422, 0.018650062382221222, -0.04127627983689308, -0.02158484235405922, 0.006559303961694241, 0.011820271611213684, 0.0171623844653368, 0.025263462215662003, 0.006326009053736925, 0.0010261591523885727, 0.01904226839542389, 0.006285435985773802, -0.042060691863298416, -0.003874722868204117, -0.017000092193484306, -0.0025932916905730963, 0.019082840532064438, -0.03711078315973282, 0.007478958927094936, -0.014105884358286858, 0.02172008715569973, 0.007722396869212389, 0.0013482074718922377, 0.008310705423355103, -0.025601571425795555, -0.02297784946858883, 0.005636267829686403, 0.010670702904462814, 0.039355821907520294, -0.01813613623380661, -0.005943946540355682, 0.005105437710881233, -0.044089339673519135, -0.006711452733725309, -0.0022179915104061365, 0.0017429490108042955, 1.0849847058125306e-05, 0.017148859798908234, 0.0008566822507418692, -0.005453689023852348, -0.017973845824599266, 0.014011213555932045, -0.02680523693561554, -0.0019627194851636887, -0.01242886669933796, 0.0023228051140904427, 0.005994662642478943, 0.016039865091443062, 0.018149660900235176, 0.020651664584875107, -0.014443992637097836, -0.012861644849181175, -0.015985766425728798, 0.02923961915075779, -0.01174588780850172, -0.012489725835621357, -0.01977258175611496, -0.002373521216213703, -0.030862538143992424, -0.01256410963833332, 0.002258564345538616, 0.009757809340953827, -0.0009915030095726252, 0.006234719883650541, -0.0068433149717748165, -0.03316167742013931, -0.03299938514828682, -0.012050185352563858, 0.003600854892283678, -0.024276185780763626, -0.007147612515836954, 0.024830684065818787, 0.018366049975156784, -0.014971441589295864, -0.027995379641652107, -0.007560105063021183, -0.023126617074012756, -0.0012721330858767033, 0.0045475587248802185, 0.006052141077816486, 0.010907378047704697, -0.0009712165337987244, -0.005223775282502174, -0.003386156167834997, 0.0036853819619864225, -0.005778273567557335, -0.052717868238687515, -0.009771334007382393, 0.005984519608318806, 0.012996888719499111, 0.0334051139652729, 0.01729762740433216, -0.00544016482308507, -0.011833795346319675, -0.0016660293331369758, 0.007729159202426672, -0.017419347539544106, 0.008161937817931175, -0.008310705423355103, -0.014403419569134712, -0.014227603562176228, 0.0065254932269454, -0.0071814232505857944, 0.004496842157095671, -0.016689032316207886, -0.006985320709645748, 0.006887269206345081, -0.013605483807623386, 0.01498496625572443, -0.021774183958768845, -0.026156069710850716, 0.005507786758244038, -0.004811283200979233, 0.005352256819605827, 0.00139892369043082, -0.03624522686004639, -0.011982562951743603, 0.001028694910928607, -0.02185533009469509, -0.0045948936603963375, 0.012232763692736626, 0.03824682906270027, -0.01879882998764515, -0.011806746944785118, -0.005034434609115124, 0.00012393787619657815, -0.014268176630139351, 0.01632387563586235, 0.0011757721658796072, -0.002490168670192361, 0.01809556409716606, 0.01352433767169714, -0.0030649530235677958, -0.008885489776730537, 0.01242886669933796, 0.0006533945561386645, 0.01632387563586235, -0.012090758420526981, 0.0010616604704409838, -0.01904226839542389, -0.030673198401927948, -0.0059067546389997005, -0.02209876850247383, 0.010528696700930595, -0.005105437710881233, 0.009385890327394009, 0.006038616877049208, -0.014498090371489525, 0.0068906499072909355, -0.000496596796438098, -0.028482254594564438, 0.0064206793904304504, -0.005396210588514805, 0.019975446164608, -0.010839756578207016, 0.0031832910608500242, 0.021273784339427948, 0.010670702904462814, 0.005379305221140385, -0.015458318404853344, -0.010278496891260147, -0.015498891472816467, 0.01660788618028164, -0.009365604259073734, -0.0012121187755838037, -0.010447550565004349, -0.02443847805261612, -0.0043649799190461636, -0.014849723316729069, -0.029050277546048164, 0.012712877243757248, 0.014660382643342018, 0.006985320709645748, -0.007107039447873831, -0.008243083953857422, -0.018460720777511597, 0.011164341121912003, 0.01681075245141983, 0.0036312846932560205, -0.0018849546322599053, -0.004104636609554291, -0.01242886669933796, 0.04395409673452377, 0.003081858390942216, 0.011177864857017994, 0.014768577180802822, -0.008155175484716892, 0.007201709784567356, 0.012841358780860901, 0.018109088763594627, -0.008418899960815907, 0.005020910408347845, 0.01324708852916956, 0.012158379890024662, 0.008750246837735176, 0.010264972224831581, 0.03129531815648079, -0.021679513156414032, -0.023329483345150948, -0.02707572467625141, 0.0224909745156765, -0.005017529241740704, 6.270855374168605e-05, -0.011394254863262177, -0.0006415607640519738, 0.015552988275885582, -0.011820271611213684, -0.0034757547546178102, -0.02408684603869915, -0.016513217240571976, -0.007844115607440472, 0.01324708852916956, 0.024546673521399498, -0.012043423019349575, -0.013531100004911423, -0.0006690321024507284, 0.008405376225709915, 0.002917876001447439, -0.012104282155632973, 0.026642944663763046, -0.0071340883150696754, -0.021801233291625977, -0.00025886428193189204, 0.0043649799190461636, 0.020448798313736916, 0.003722574096173048, -0.022369254380464554, 0.02112501487135887, 0.029023228213191032, -0.011948752216994762, 0.0305920522660017, -0.004946526605635881, 0.0024039510171860456, -0.040843501687049866, 0.013355283997952938, -0.000500823138281703, -0.024235613644123077, 0.012733164243400097, 0.005622743628919125, -0.025006501004099846, -0.009467036463320255, 0.013578435406088829, -0.008141651749610901, -0.01629682630300522, -0.02673761546611786, -0.002508764620870352, 0.02380283549427986, 0.022261060774326324, -0.011793222278356552, -0.01249648816883564, 0.007086752913892269, -0.005744462367147207, 0.0024479052517563105, 0.029401909559965134, -0.012746687978506088, -0.015796426683664322, 0.044116389006376266, 0.0245060995221138, 0.005859419237822294, -0.018190234899520874, 0.005433402955532074, -0.00845947302877903, 0.022261060774326324, 0.014809150248765945, -0.024005699902772903, -0.016689032316207886, 0.005095294211059809, -0.0001916652254294604, -0.00654239859431982, -0.004195925779640675, -0.014132932759821415, 0.005419878289103508, -0.0053150649182498455, 0.02627778798341751, 0.02161189168691635, -0.029645347967743874, 0.008533856831490993, -0.002875612350180745, 0.023410629481077194, -0.004719994030892849, -0.034757547080516815, 0.004385266453027725, 0.02961830049753189, -0.003176528960466385, -0.012557347305119038, 0.01155654713511467, -0.005856038071215153, -0.03808453679084778, 0.0002954222436528653, -0.027454406023025513, 0.029266666620969772, 0.01420055516064167, -0.00703941797837615, 0.005118961911648512, 0.01990782469511032, -0.001217190409079194, 0.002921256935223937, 0.008662338368594646, -0.022950801998376846, -0.013037461787462234, -0.008040218614041805, 0.010576032102108002, 0.029888786375522614, -0.008973398245871067, 0.013125370256602764, 0.02614254504442215, 0.0021148682571947575, -0.005670078564435244, 0.013078034855425358, 0.010264972224831581, 0.012334195896983147, -0.01712181232869625, 0.003345583099871874, 0.022288108244538307, -0.011637692339718342, -0.0051933457143604755, -0.02346472628414631, 0.0004577143117785454, 0.022720888257026672, -0.01467390637844801, -0.01979963108897209, -0.009534657932817936, -0.0012577634770423174, -0.017081238329410553, 0.009270933456718922, -0.0027420595288276672, 0.0034436346031725407, -0.011603881604969501, -0.018785305321216583, 0.011712076142430305, -0.02359996922314167, 0.008229559287428856, 0.012936029583215714, 0.005125724244862795, -0.020584043115377426, -0.011799984611570835, 0.028455207124352455, 0.0034199669025838375, -0.012854883447289467, 0.004375123418867588, -0.014579236507415771, 0.011076432652771473, 0.016026340425014496, -0.004280453082174063, -0.025087647140026093, -0.003404752118512988, -0.009622566401958466, 0.009595518000423908, 0.0014006142737343907, -0.004381885286420584, -0.0020100546535104513, -0.002081057522445917, 0.010623367503285408, -0.019407425075769424, 0.007580391131341457, 0.004297358449548483, -0.025141743943095207, 0.013132131658494473, 0.007147612515836954, 0.008838154375553131, -0.0053083025850355625, -0.007999645546078682, -0.0033151532988995314, 0.011820271611213684, 0.014105884358286858, -0.01979963108897209, -0.009020733647048473, 0.22461220622062683, -0.011820271611213684, 0.008283657021820545, 0.045063093304634094, -0.024452002719044685, -0.0036211414262652397, 0.02474953792989254, -0.004344693385064602, -0.00497695617377758, -0.016066912561655045, -0.023897504433989525, 0.006559303961694241, 0.0012780498946085572, 0.007850877940654755, 0.0035704250913113356, -0.027751941233873367, -0.022288108244538307, -0.0030497382394969463, -0.010711275972425938, 0.0035095657221972942, 0.00995391234755516, -0.0034216574858874083, 0.0012256430927664042, -0.013673105277121067, 0.03316167742013931, 0.020651664584875107, 0.0004610953910741955, 0.039004191756248474, -0.001438651466742158, 0.024181516841053963, -0.005947327706962824, 0.00010988524445565417, 0.006001424975693226, 0.002885755617171526, -0.0023954983334988356, 0.0027082485612481833, -0.0175816398113966, 0.0015383934369310737, 0.014159982092678547, 0.014619809575378895, 0.007965834811329842, -0.014038262888789177, -0.0008444258710369468, -0.005014148075133562, 0.01904226839542389, 0.04298034682869911, -0.008560906164348125, -0.005744462367147207, 0.01572880521416664, 0.02722449228167534, -0.016283303499221802, -0.004517128691077232, 0.018771780654788017, 0.031863339245319366, -0.00048814405454322696, -0.008445949293673038, 0.00988629087805748, 0.011096719652414322, 0.00988629087805748, 0.004121541976928711, -0.016337400302290916, 0.013078034855425358, -0.0006829790654592216, 0.025804435834288597, -0.012908980250358582, 0.04546882212162018, -0.016161583364009857, 0.010420502163469791, 0.022504497319459915, -0.001377791864797473, 0.018474245443940163, 0.00261864997446537, 0.006430822424590588, -0.009216836653649807, -0.02959125116467476, -0.012320672161877155, 0.025642145425081253, 0.012726401910185814, -0.006819647271186113, 0.01764926128089428, -0.01778450421988964, 0.0012078924337401986, 0.007871164940297604, 0.021354928612709045, -0.005785035435110331, -0.03565015271306038, 0.010122966952621937, -0.007161136716604233, -0.010264972224831581, -0.016242729499936104, 0.001871430198661983, -0.026669993996620178, -0.019975446164608, -0.012557347305119038, -0.009737523272633553, 0.0151066854596138, -0.008229559287428856, 0.0016381354071199894, -0.013970641419291496, 0.0021486792247742414, -0.04249347001314163, 0.04292624816298485, 0.008574429899454117, 0.02813062258064747, 0.003993060905486345, 0.02614254504442215, 0.016053389757871628, 0.01355814840644598, -0.007032655645161867, 0.0011825342662632465, -0.010663940571248531, -0.005463832523673773, 0.006058903411030769, 0.0028181339148432016, 0.005680222064256668, -0.0011554856318980455, 0.008128127083182335, -0.025114694610238075, -9.218737977789715e-05, 0.004949907772243023, -0.011150816455483437, -0.021814756095409393, -0.029266666620969772, -0.002503693103790283, 0.002938162302598357, 0.0096090417355299, -0.01844719611108303, 0.004537415225058794, 0.007932024076581001, -0.022788509726524353, 0.025574522092938423, -0.012888694182038307, -0.024343807250261307, -0.017081238329410553, -0.02098977193236351, -0.029753543436527252, 0.015079637058079243, -0.02384340763092041, -0.0047166128642857075, -0.0031494800932705402, 0.019204558804631233, -0.0007117182831279933, -0.005504405591636896, -0.004929621238261461, 0.007411336991935968, -0.0016795536503195763, 0.010860043577849865, -0.004243261180818081, -0.03278299421072006, -0.014457517303526402, -0.008472997695207596, -0.00482142623513937, -0.006471395492553711, -0.03532557189464569, 0.04195249453186989, -0.010427264496684074, -0.042168885469436646, -0.022017622366547585, 0.0034520872868597507, 0.009690187871456146, -0.04511719197034836, 0.023194238543510437, 0.026291312649846077, -0.035974737256765366, -0.010718037374317646, -0.02328890934586525, -0.17408527433872223, 0.02370816469192505, 0.010846518911421299, 0.0011428064899519086, 0.01709476299583912, -0.0027775608468800783, 0.019921349361538887, 0.012023136019706726, -0.01928570494055748, -0.009196549654006958, 0.025385182350873947, 0.009615804068744183, -0.031241219490766525, 0.01873120851814747, 0.011799984611570835, -0.001993149286136031, -0.028915034607052803, 0.008540619164705276, -0.0005075852968730032, 0.017243530601263046, 0.016229204833507538, -0.01124548725783825, 0.001323694596067071, -0.012699353508651257, -0.010447550565004349, 0.006738501135259867, -0.025263462215662003, 0.012117806822061539, -0.010724799707531929, -0.015147258527576923, -0.00842566229403019, -0.0025155269540846348, 0.011394254863262177, 0.013071272522211075, 0.007167899049818516, 0.004953288938850164, -0.012016374617815018, -0.006552541628479958, 0.0030125463381409645, 0.03702963516116142, -0.010014772415161133, 0.022774985060095787, -0.0026085067074745893, 0.010001247748732567, 0.013842159882187843, 0.030889587476849556, 0.009980961680412292, 0.007364002056419849, -0.005849276203662157, -0.016472643241286278, -0.015404220670461655, -0.025263462215662003, 0.003151170676574111, -0.011272535659372807, 0.03080844134092331, -0.006684403866529465, -0.02468191646039486, 0.025074122473597527, -0.00964285247027874, -0.012760212644934654, 0.004638847894966602, -0.045198336243629456, -0.005859419237822294, 0.007790018804371357, -0.011874368414282799, -0.005998043809086084, -0.005102056544274092, -0.009406177327036858, -0.0200160201638937, -0.0053691621869802475, -0.009412938728928566, -0.00898016057908535, 0.010799183510243893, -0.018014417961239815, 0.023938078433275223, -0.00827013235539198, -0.031592853367328644, 0.009446750394999981, -0.012699353508651257, 0.007661537267267704, -0.0009188096737489104, 0.025452803820371628, 0.0035433764569461346, 0.002476644469425082, 0.016594363376498222, 0.005159534979611635, 0.008310705423355103, 0.028292914852499962, 0.0028553258161991835, -0.019988970831036568, 0.018325477838516235, -0.0011411160230636597, -0.032701849937438965, -0.02468191646039486, -0.017378773540258408, 0.008019932545721531, 0.010413739830255508, -0.00504795927554369, -0.0050783888436853886, -0.011935228481888771, 0.011360444128513336, -0.0017632355447858572, -0.0305920522660017, 0.02085452899336815, 0.020529944449663162, 0.009412938728928566, -0.0044866991229355335, 0.0009162739152088761, 0.007303142454475164, -0.01094795111566782, -0.031971532851457596, 0.023478250950574875, 0.017216481268405914, 0.0010481361532583833, 0.01667550764977932, 0.027549076825380325, -0.0053691621869802475, -0.016513217240571976, -0.00894634984433651, -0.011610643938183784, 0.07438385486602783, 0.01583699882030487, 0.0028908271342515945, -0.006937985308468342, 0.006894031073898077, -0.009230360388755798, -0.12507307529449463, -0.005940565373748541, -0.00758715346455574, -0.005095294211059809, -0.0022196818608790636, -0.009352079592645168, 0.003634665859863162, -0.002868850249797106, -0.018284903839230537, -0.0005058947717770934, 0.0025459565222263336, -0.035623107105493546, -0.029996981844305992, -0.010704513639211655, -0.009203311987221241, 0.007086752913892269, -0.002829967765137553, -0.00988629087805748, -0.018433673307299614, 0.030024029314517975, 0.0039051524363458157, -0.0031714572105556726, 0.008019932545721531, -0.020245933905243874, 0.012996888719499111, 0.00995391234755516, -0.03711078315973282, 0.02136845327913761, 0.018109088763594627, 0.00576813006773591, -0.0087908199056983, 0.010386691428720951, -0.0012746688444167376, -0.006681022699922323, 0.008053743280470371, -0.02234220691025257, 0.0004843403585255146, -0.014241127297282219, 0.030970733612775803, -0.02273441106081009, 0.012861644849181175, 0.0015147258527576923, -0.017257055267691612, -0.019867252558469772, 0.0020878196228295565, -0.003026070538908243, -0.023167191073298454, 0.01718943379819393, 0.009683425538241863, -0.009771334007382393, -0.028833888471126556, 0.013334996998310089, -0.018393099308013916, -0.022883180528879166, 0.019380375742912292, -0.002168965758755803, 0.026899907737970352, 0.02468191646039486, 0.0060217115096747875, 0.011482162401080132, 0.007242282852530479, 0.011184627190232277, -0.010515172965824604, 0.02673761546611786, 0.009325031191110611, 0.015404220670461655, -0.010778897441923618, -0.014782100915908813, 0.01493086852133274, -0.018812354654073715, 0.015931669622659683, 0.015755852684378624, -0.0018308572471141815, 0.02840111032128334, 0.009859242476522923, 0.007837354205548763, -0.02328890934586525, -0.017811553552746773, -0.00022695529332850128, -0.026156069710850716, -0.00425678538158536, -0.0061975279822945595, 0.021449599415063858, -0.012760212644934654, 0.0416279137134552, 0.0037462415639311075, -0.0104069784283638, 0.021814756095409393, 0.009325031191110611, 0.0022450401447713375, 0.0071340883150696754, 0.019813155755400658, 0.011252248659729958, -0.03478459641337395, -0.008074029348790646, 0.010508410632610321, 0.0016203847480937839, -0.005264348350465298, 0.024722490459680557, -0.023897504433989525, -0.041925448924303055, -0.018704159185290337, -0.01917751133441925, 0.017554590478539467, -0.013416143134236336, -0.017689833417534828, -0.02373521216213703, -0.016486167907714844, 0.010400216095149517, 0.00028739217668771744, -0.011705314740538597, 0.006549160461872816, -0.0017074476927518845, 0.023721689358353615, -0.0024124037008732557, -0.018257856369018555, -0.01855539157986641, -0.010643653571605682, 0.013078034855425358, -0.014484565705060959, -0.003573806257918477, 0.003485898021608591, 0.008364803157746792, 0.0026152688078582287, 0.023816358298063278, -0.0015476914122700691, -0.009068068116903305, 0.011421303264796734, -0.014633333310484886, 0.015809951350092888, -0.01498496625572443, 0.0036955252289772034, -0.006123144179582596, -0.010860043577849865, -0.01587757281959057, 0.0024665012024343014, -0.008642052300274372, -0.03486574441194534, 0.0033607978839427233, -0.0096090417355299, 0.01979963108897209, 0.006038616877049208, 0.0012222620425745845, -0.03202563151717186, 0.0022805414628237486, -0.025452803820371628, 0.004665896762162447, -0.004703088663518429, -0.03678619861602783, 0.01629682630300522, 0.004182401578873396, 0.009399414993822575, 0.005010767374187708, 0.003201887011528015, -0.021327881142497063, -0.03859845921397209, -0.007932024076581001, -0.01601281575858593, 0.005626124329864979, 0.0016711009666323662, -0.005244061816483736, -0.0035771874245256186, 0.035596057772636414, -0.0060217115096747875, 0.014362846501171589, -0.004277071915566921, 0.016134535893797874, 0.012827834114432335, 0.0067892177030444145, 0.03421657532453537, -0.008587954565882683, -0.031592853367328644, -0.012239526025950909, 0.006579590495675802, -0.015891097486019135, 0.008087554015219212, 0.0006817111279815435, 0.012956315651535988, -0.02534460835158825, -0.011908179149031639, 5.504828004632145e-05, 0.038057487457990646, 0.0016668746247887611, -0.022328682243824005, -0.023572921752929688, -0.0017497112276032567, -0.0025611715391278267, 0.008959873579442501, -0.024235613644123077, 0.01517430692911148, -0.018677109852433205, 0.003972774371504784, -0.012692591175436974, 0.009676664136350155, -0.0025814580731093884, 0.00677907420322299, -0.011509211733937263, 0.014876771718263626, -0.001248465385288, -0.003803719999268651, 0.021841805428266525, 0.041925448924303055, 0.020732810720801353, -0.003496041288599372, -0.0002983807062264532, -0.028536353260278702, -0.0327288992702961, 0.0009762881090864539, -0.02746793068945408, -0.032837092876434326, -0.010325832292437553, 0.016621410846710205, 0.014322273433208466, 0.011455113999545574, 0.03521737456321716, 0.0007045334787108004, 0.011996087618172169, 0.03175514563918114, -0.009250647388398647, -0.04279100522398949, -0.018825877457857132, 0.0053150649182498455, 0.012611445039510727, -0.022261060774326324, 0.014768577180802822, 0.004348074551671743, 0.03770585358142853, 0.005849276203662157, 0.012395055964589119, -0.016689032316207886, -0.0004378504236228764, -0.012942790985107422, 0.011022334918379784, -0.01994839869439602, -0.023383580148220062, -0.008209273219108582, -0.01862301304936409, -0.010393453761935234, -0.008993684314191341, 0.036948490887880325, -0.03624522686004639, 0.018704159185290337, -0.00046997074969112873, -0.014903820119798183, 0.01900169439613819, 0.007370763923972845, 0.02547985315322876, 0.011292821727693081, -0.006532255094498396, -0.029942883178591728, 0.009723998606204987, 0.01367986761033535, -0.02004306949675083, 0.017351726070046425, -0.020313555374741554, -0.003399680368602276, 0.013220040127635002, 0.004750423599034548, 0.011076432652771473, -0.00559569476172328, -0.016202157363295555, 0.031214172020554543, 0.01649969257414341, 0.01364605687558651, 0.004402171820402145, 0.014903820119798183, -0.02579091303050518, -0.0036752386949956417, -0.019191036000847816, 0.014430468901991844, -0.0032137208618223667, 0.011678265407681465, -0.0038003388326615095, -5.964444426354021e-05, -0.018677109852433205, -0.0036143793258816004, 0.009142451919615269, -0.00814841315150261, 0.003063262440264225, 0.006278673652559519, -0.004361598752439022, -0.0030412855558097363, 0.023789310827851295, -0.02074633352458477, -0.049444980919361115, 0.01778450421988964, 0.011299584060907364, -0.01698656752705574, 0.002779251430183649, -0.008202510885894299]"))')
'''

#add a vector embedding to our database
#add a speech/transcript keypoint to our database--this is called per keypoint per speech per speaker
def add_vector(speaker_id, transcript_id, text, vector):
    table = 'speakervectors'
    ADD_VECTOR_SQL = f'INSERT INTO speakervectors (speaker_id, transcript_id, text, vector) VALUES ("{speaker_id}", "{transcript_id}", "{text}", JSON_ARRAY_PACK("{vector}"));'
    with conn:
        with conn.cursor() as cur:
            cur.execute(ADD_VECTOR_SQL)

#get the similar embeddings to the given vector embedding
#we will be matching a given new set of keypoints with the existing keypoints
#After finding the k most smilar keypoints in the model speaker's speech, we'll
#return those similar keypoints. We simply are matching the linguistic scenarios
#here so we can pass those into hume later for the emotional cues and expressions that we expect
def search(embedding):
    SEARCH_SQL = f'select text, dot_product(vector, JSON_ARRAY_PACK("${embedding}")) as score from my speakervectors order by score desc limit 5;'
    with conn:
        with conn.cursor() as cur:
            cur.execute(SEARCH_SQL)

            