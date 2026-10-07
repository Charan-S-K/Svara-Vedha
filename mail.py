import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import smtplib
import secrets

load_dotenv()

def generator():
    return (secrets.randbelow(900000)+100000)

def send_mail(to):
    ser = 'smtp.gmail.com'
    port = 587
    email = os.getenv("EMAIL_HOST_USER")
    password = os.getenv("EMAIL_HOST_PASSWORD")
    try:
        msg=MIMEMultipart()
        msg['From']=email
        msg['to']=to
        msg['Subject']="Your Svara Bava Verification Code"
        otp = generator()
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Svara Bava - Email Verification</title>
</head>

<body style="
    margin: 0;
    padding: 0;
    background-color: #f3f6fb;
    font-family: Arial, Helvetica, sans-serif;
    color: #172033;
">

    <table width="100%" cellpadding="0" cellspacing="0" border="0"
        style="background-color: #f3f6fb; padding: 40px 15px;">

        <tr>
            <td align="center">

                <!-- MAIN CONTAINER -->
                <table width="100%" cellpadding="0" cellspacing="0" border="0"
                    style="
                        max-width: 680px;
                        background-color: #ffffff;
                        border-radius: 18px;
                        overflow: hidden;
                        box-shadow: 0 8px 30px rgba(15, 35, 75, 0.12);
                    ">

                    <!-- ================= HEADER ================= -->
                    <tr>
                        <td style="
                            background: linear-gradient(135deg, #173b91, #234eb4);
                            padding: 42px 48px 48px 48px;
                            color: #ffffff;
                        ">

                            <!-- BRAND -->
                            <table width="100%" cellpadding="0" cellspacing="0">
                                <tr>

                                    <td width="60" valign="middle">

                                        <!-- LOGO PLACEHOLDER -->
                                        <div style="
                                            width: 46px;
                                            height: 46px;
                                            background-color: #3d8cff;
                                            border-radius: 12px;
                                            text-align: center;
                                            line-height: 46px;
                                            color: #ffffff;
                                            font-size: 20px;
                                            font-weight: bold;
                                        ">
                                            SB
                                        </div>

                                    </td>

                                    <td valign="middle">

                                        <div style="
                                            font-size: 25px;
                                            font-weight: bold;
                                            letter-spacing: -0.5px;
                                        ">
                                            Svara Bava
                                        </div>

                                        <div style="
                                            margin-top: 4px;
                                            font-size: 13px;
                                            color: #dbe7ff;
                                        ">
                                            Voice for a Better You
                                        </div>

                                    </td>

                                </tr>
                            </table>


                            <!-- HEADER CONTENT -->
                            <table width="100%" cellpadding="0" cellspacing="0"
                                style="margin-top: 55px;">

                                <tr>

                                    <td width="58%" valign="top">

                                        <div style="
                                            font-size: 38px;
                                            line-height: 1.12;
                                            font-weight: bold;
                                            color: #ffffff;
                                        ">
                                            Verify Your<br>
                                            Email Address
                                        </div>

                                        <div style="
                                            margin-top: 22px;
                                            font-size: 16px;
                                            line-height: 1.6;
                                            color: #dce7ff;
                                        ">
                                            Take the next step towards a secure
                                            and personalized experience with
                                            Svara Bava.
                                        </div>

                                    </td>

                                    <!-- SECURITY ICON -->
                                    <td width="42%" align="center" valign="middle">

                                        <div style="
                                            width: 125px;
                                            height: 125px;
                                            border-radius: 50%;
                                            background-color: rgba(255,255,255,0.10);
                                            border: 1px solid rgba(255,255,255,0.20);
                                            text-align: center;
                                            line-height: 125px;
                                            font-size: 55px;
                                        ">
                                            🔒
                                        </div>

                                    </td>

                                </tr>

                            </table>

                        </td>
                    </tr>


                    <!-- ================= CONTENT ================= -->
                    <tr>
                        <td style="
                            padding: 45px 48px 35px 48px;
                            background-color: #ffffff;
                        ">

                            <!-- GREETING -->

                            <div style="
                                font-size: 28px;
                                font-weight: bold;
                                color: #172033;
                            ">
                                Hello,
                            </div>

                            <div style="
                                margin-top: 15px;
                                font-size: 16px;
                                line-height: 1.65;
                                color: #5b6780;
                            ">
                                Thank you for choosing
                                <strong style="color: #172033;">
                                    Svara Bava
                                </strong>.
                                Please use the verification code below
                                to confirm your email address and continue.
                            </div>


                            <!-- OTP BOX -->

                            <table width="100%" cellpadding="0" cellspacing="0"
                                style="
                                    margin-top: 30px;
                                    background-color: #eef5ff;
                                    border: 1px solid #d7e6ff;
                                    border-radius: 12px;
                                ">

                                <tr>
                                    <td align="center" style="padding: 22px;">

                                        <div style="
                                            font-size: 12px;
                                            font-weight: bold;
                                            letter-spacing: 3px;
                                            color: #61708c;
                                            margin-bottom: 15px;
                                        ">
                                            YOUR VERIFICATION CODE
                                        </div>

                                        <!-- OTP -->

                                        <div style="
                                            display: inline-block;
                                            background-color: #ffffff;
                                            border-radius: 10px;
                                            padding: 16px 30px;
                                            font-size: 34px;
                                            font-weight: bold;
                                            letter-spacing: 10px;
                                            color: #2050b8;
                                            border: 1px solid #e3ebf8;
                                        ">
                                            {otp}
                                        </div>

                                    </td>
                                </tr>

                            </table>


                            <!-- EXPIRY -->

                            <div style="
                                margin-top: 18px;
                                text-align: center;
                                font-size: 14px;
                                color: #5b6780;
                            ">
                                This code will expire in
                                <strong style="color: #172033;">
                                    10 minutes
                                </strong>
                                for your security.
                            </div>


                            <!-- VERIFY BUTTON -->

                            <table width="100%" cellpadding="0" cellspacing="0"
                                style="margin-top: 25px;">

                                <tr>
                                    <td align="center">

                                        <a href="#"
                                            style="
                                                display: inline-block;
                                                background-color: #173f96;
                                                color: #ffffff;
                                                text-decoration: none;
                                                font-size: 16px;
                                                font-weight: bold;
                                                padding: 15px 38px;
                                                border-radius: 30px;
                                            ">
                                            Verify Email&nbsp;&nbsp; →
                                        </a>

                                    </td>
                                </tr>

                            </table>


                            <!-- DIVIDER -->

                            <div style="
                                margin-top: 30px;
                                border-top: 1px solid #e5e9f0;
                            ">
                            </div>


                            <!-- SECURITY MESSAGE -->

                            <div style="
                                margin-top: 20px;
                                text-align: center;
                                font-size: 13px;
                                line-height: 1.6;
                                color: #68758c;
                            ">
                                If you did not request this verification code,
                                you can safely ignore this email.
                            </div>

                        </td>
                    </tr>


                    <!-- ================= FOOTER ================= -->

                    <tr>
                        <td style="
                            background-color: #eef5ff;
                            padding: 28px 48px;
                        ">

                            <table width="100%" cellpadding="0" cellspacing="0">

                                <tr>

                                    <!-- BRAND -->

                                    <td valign="top">

                                        <div style="
                                            font-size: 18px;
                                            font-weight: bold;
                                            color: #173f96;
                                        ">
                                            ◆ Svara Bava
                                        </div>

                                        <div style="
                                            margin-top: 4px;
                                            font-size: 12px;
                                            color: #64738d;
                                        ">
                                            Voice for a Better You
                                        </div>

                                    </td>


                                    <!-- LINKS -->

                                    <td align="right" valign="top">

                                        <div style="
                                            font-size: 12px;
                                            color: #5c6a82;
                                        ">
                                            Privacy
                                            &nbsp;&nbsp;•&nbsp;&nbsp;
                                            Terms
                                            &nbsp;&nbsp;•&nbsp;&nbsp;
                                            Support
                                        </div>

                                    </td>

                                </tr>

                            </table>


                            <!-- COPYRIGHT -->

                            <div style="
                                margin-top: 22px;
                                font-size: 12px;
                                color: #697791;
                            ">
                                © 2026 Svara Bava. All rights reserved.
                            </div>

                        </td>
                    </tr>

                </table>

            </td>
        </tr>

    </table>

</body>
</html>
"""

        msg.attach(MIMEText(html,'html'))

        with smtplib.SMTP(ser,port) as server:
            server.starttls()
            server.login(email,password)
            server.sendmail(email,to,msg.as_string())
            print('email sent successfully')
    except Exception as e:
        print(f"Error in sending email {e}")
    return otp

if __name__=='__main__':
    send_mail('www.c.tej2000@gmail.com')