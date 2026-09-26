# Breast Cancer Lab

یک آزمایشگاه کوچک و بازتولیدپذیر برای مقایسهٔ مدل کلاسیک با شبکهٔ عصبی PyTorch روی دو مجموعه‌دادهٔ مستقل Wisconsin. آموزش شبکه در صورت دسترسی از CUDA/GPU استفاده می‌کند و برای اجرا روی CPU هم قابل انتخاب است.

> این پروژه آموزشی و پژوهشی است؛ ابزار تشخیص پزشکی نیست و برای تصمیم‌گیری بالینی تأیید نشده است.

## در یک نگاه

- آموزش و ارزیابی جداگانه روی دو دیتاست؛ ویژگی‌های ناهمسان با هم ادغام نمی‌شوند.
- شبکهٔ fully connected با ساختار `input → 64 → 32 → 1`، ReLU، dropout، AdamW و توقف زودهنگام.
- انتخاب خودکار GPU با PyTorch؛ امکان اجبار به CUDA یا CPU از خط فرمان.
- اعتبارسنجی طبقه‌بندی‌شدهٔ ۵-fold، مجموعهٔ آزمون مستقل ۲۰٪ و گزارش precision، recall، F1 و ROC-AUC در کنار accuracy.
- پیش‌پردازش در هر fold فقط از دادهٔ آموزش یاد گرفته می‌شود؛ ۱۶ مقدار گمشدهٔ دیتاست UCI با median همان بخش آموزش جایگزین می‌شود.
- نمودارهای ROC، ماتریس خطا، ساختار نورون‌ها و checkpointهای قابل استفادهٔ مجدد.

## نتایج نمونه

یک اجرای بازتولیدپذیر با seed ثابت؛ اعداد آزمون متعلق به همان split هستند و تضمین عملکرد روی دادهٔ جدید نیستند.

| دیتاست | مدل | CV accuracy | آزمون accuracy | آزمون recall بدخیم | آزمون ROC-AUC |
|---|---|---:|---:|---:|---:|
| Wisconsin Diagnostic (569 نمونه) | Logistic regression | 97.8% | 99.1% | 97.6% | 99.1% |
| Wisconsin Diagnostic (569 نمونه) | PyTorch + CUDA | 97.4% | 95.6% | 97.6% | 99.7% |
| Wisconsin Original (699 نمونه) | Logistic regression | 96.6% | 96.4% | 93.8% | 99.8% |
| Wisconsin Original (699 نمونه) | PyTorch + CUDA | 97.1% | 97.1% | 95.8% | 99.8% |

شبکه در دیتاست دوم برندهٔ CV شد؛ روی دیتاست Diagnostic، رگرسیون لجستیک بهتر ماند. انتخاب مدل برای هر دیتاست مستقل است و صرفاً بر اساس CV بخش آموزش انجام می‌شود.

## نمودارها

| Wisconsin Diagnostic | Wisconsin Original |
|---|---|
| ROC و ماتریس خطا: [مشاهده](outputs/model_evaluation.png) | ROC و ماتریس خطا: [مشاهده](outputs/model_evaluation_original.png) |
| ساختار شبکه: [مشاهده](outputs/neural_network.png) | ساختار شبکه: [مشاهده](outputs/neural_network_original.png) |

## اجرا در PowerShell

Python را با مسیر کامل اجرا کنید؛ لازم نیست در `PATH` باشد. مسیر زیر با نصب فعلی این سیستم سازگار است؛ اگر Python جای دیگری نصب شده، مقدار `$Python` را عوض کنید.

```powershell
$Python = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $Python --version
& $Python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& $Python -m pip install -r requirements.txt
& $Python cancer.py --device auto
```

برای اجبار به GPU از `--device cuda` و برای اجرای CPU از `--device cpu` استفاده کنید. اگر PyTorch/CUDA هنوز نصب نیست، wheel مناسب کارت گرافیک و نسخهٔ CUDA را از [راهنمای رسمی نصب PyTorch](https://pytorch.org/get-started/locally/) انتخاب کنید؛ نصب PyTorch موجود در این محیط دست‌کاری نشده است.

فایل `cc.csv` کنار `cancer.py` و دیتاست دوم در `data/wisconsin_original.data` قرار دارد. اجرای کامل چند fold شبکهٔ عصبی دارد و از اجرای یک مدل خطی زمان بیشتری می‌برد.

## خروجی‌ها

پوشهٔ نادیده‌گرفته‌شدهٔ `outputs/` هنگام اجرا ساخته می‌شود:

- `model_metrics.csv`: معیارهای CV و آزمون برای هر چهار ترکیب دیتاست/مدل.
- `model_evaluation*.png`: منحنی ROC و ماتریس خطای مدل منتخب هر دیتاست.
- `neural_network*.png`: نورون‌ها و وزن‌های آموخته‌شدهٔ شبکهٔ PyTorch.
- `*_logistic.joblib`: pipeline رگرسیون لجستیک شامل imputer و scaler.
- `*_pytorch_checkpoint.pt`: وزن‌های شبکه، ترتیب ویژگی‌ها و آمار پیش‌پردازش.

برای بارگذاری checkpoint از `load_torch_checkpoint` و برای پیش‌بینی روی DataFrame هم‌ساختار از `predict_checkpoint` در `cancer.py` استفاده کنید. احتمال خروجی، احتمال مدل است و کالیبراسیون بالینی محسوب نمی‌شود.

## داده و ارجاع

دیتاست افزوده‌شده، Wisconsin Breast Cancer (Original) با ۶۹۹ نمونه و ۹ ویژگی است. منبع رسمی: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)، DOI: [10.24432/C5HP4Z](https://doi.org/10.24432/C5HP4Z).

فایل Diagnostic موجود (`cc.csv`) متناظر با Wisconsin Breast Cancer (Diagnostic)، ۵۶۹ نمونه و ۳۰ ویژگی است؛ منبع: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)، DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). هر دو مجموعه‌داده با مجوز [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) منتشر شده‌اند. چون schema متفاوت است، هرکدام مدل و ارزیابی مستقل دارند؛ پیش‌پردازش هرگز روی مجموعهٔ آزمون fit نمی‌شود.
