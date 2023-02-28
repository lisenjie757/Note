# 作业一、LED灯和按键
## 1. 实验要求

- 在Alios Things操作系统下实现按键开关LED灯
## 2. 实验环境

- 硬件平台：HaaS100 开发板，硬件规格：[link](https://help.aliyun.com/document_detail/184186.html)
- 软件环境：Alios Things操作系统，源码仓库：[link](https://github.com/alibaba/AliOS-Things?spm=5176.22654207.J_5253785160.2.25936165YSFZPy)
- 开发平台：VSCode haas-studio extension
## 3. 实验过程

1. 打开HaaS Studio，选择AliOS Things开发（C），选择硬件类型haas100，选择解决方案helloword简单示例，导入代码

![屏幕截图_20221227_142836.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672123432123-f7fbdfd0-5e23-40d6-84a8-fd359357f8de.png#averageHue=%23947852&clientId=ub6cf6559-96d1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=348&id=u048f9e8e&name=%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_20221227_142836.png&originHeight=1389&originWidth=2570&originalType=binary&ratio=1&rotation=0&showTitle=false&size=214417&status=done&style=none&taskId=u765e98aa-d8dc-4b6b-8a2b-ad3671b4928&title=&width=643)

2. 将`solutions/hello_world/helloworld.c`主程序的改为如下代码
```c
/*
* Copyright (C) 2015-2020 Alibaba Group Holding Limited
*/

#include "aos/init.h"
#include "board.h"
#include <aos/errno.h>
#include <aos/kernel.h>
#include <k_api.h>
#include <stdio.h>
#include <stdlib.h>
#include "led.h"
#include "key.h"

void key_short_pressed()
{
    static led_e state = LED_OFF;
    if(state == LED_ON){
    state = LED_OFF;
    led_switch(1, state);

	}else{
    state = LED_ON;
    led_switch(1, state);
	}
}

void key_long_pressed()
{
    static led_e state = LED_OFF;
    if(state == LED_ON){
    state = LED_OFF;
    led_switch(2, state);

	}else{
    state = LED_ON;
    led_switch(2, state);
	}
}

void key_test_init()
{
    key_cfg_t cfg;
    cfg.short_press_handler = key_short_pressed;
    cfg.long_press_handler  = key_long_pressed;
    cfg.long_press_min_ms   = 3000;
    cfg.short_press_max_ms  = 1000;
    key_init(&cfg);
}

int application_start(int argc, char *argv[])
{   
    key_test_init();
}

```

3. 点击下方编译并烧录
## 4. 实验结果

1. 短按亮灭灯1

![](https://i.postimg.cc/GmkDtxBP/ezgif-com-gif-maker-1.gif#crop=0&crop=0&crop=1&crop=1&id=Miv4K&originHeight=180&originWidth=180&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

2. 长按亮灭灯2

![](https://i.postimg.cc/XYD4dpr2/ezgif-com-gif-maker.gif#crop=0&crop=0&crop=1&crop=1&id=boRoc&originHeight=180&originWidth=180&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
# 作业二、阿里云物联网平台物模型
## 0. 物模型的概念

- 物模型是阿里云物联网平台为产品定义的数据模型，用于描述产品的功能。
- 物模型是物理空间中的实体（如传感器、车载装置、楼宇、工厂等）在云端的数字化表示，从属性、服务和事件三个维度，分别描述了该实体是什么、能做什么、可以对外提供哪些信息。定义了物模型的这三个维度，即完成了产品功能的定义。
- 物联网平台支持为产品定义多组功能（属性、服务和事件）。一组功能定义的集合，就是一个物模型模块。多个物模型模块，彼此互不影响。
- 物模型模块功能，解决了工业场景中复杂的设备建模，便于在同一产品下，开发不同功能的设备。例如，电暖扇产品的功能属性有**电源开关**、**档位（高、中、低）**和**室内温度**，可以在一个模块添加前2个属性，在另一个模块添加3个属性，然后分别在不同设备端，针对不同物模型模块功能进行开发。此时，该产品下不同设备就可以实现不同功能。

| **功能类型** | **说明** |
| -----------| -------- |
| 属性（Property） | 用于描述设备运行时具体信息和状态。例如，环境监测设备所读取的当前环境温度、智能灯开关状态、电风扇风力等级等。属性可分为读写和只读两种类型。读写类型支持读取和设置属性值，只读类型仅支持读取属性值。 |
| 服务（Service） | 指设备可供外部调用的指令或方法。服务调用中可设置输入和输出参数。输入参数是服务执行时的参数，输出参数是服务执行后的结果。相比于属性，服务可通过一条指令实现更复杂的业务逻辑，例如执行某项特定的任务。服务分为异步和同步两种调用方式。 |
| 事件（Event） | 设备运行时，主动上报给云端的信息，一般包含需要被外部感知和处理的信息、告警和故障。事件中可包含多个输出参数。例如，某项任务完成后的通知信息；设备发生故障时的温度、时间信息；设备告警时的运行状态等。事件可以被订阅和推送。 |

## 1. 实验要求

- 在阿里云物联网平台上创建一个物模型
- 在Alios Things操作系统下将HaaS100连接至阿里云，实现属性上报、事件上报和服务调用
## 2. 实验环境

- 硬件平台：HaaS100 开发板，硬件规格：[link](https://help.aliyun.com/document_detail/184186.html)
- 软件环境：Alios Things操作系统，源码仓库：[link](https://github.com/alibaba/AliOS-Things?spm=5176.22654207.J_5253785160.2.25936165YSFZPy)
- 开发平台：VSCode haas-studio extension
- 阿里云物联网平台，网址：[link](https://iot.console.aliyun.com/product)   
## 3. 实验过程

1. 打开HaaS Studio，选择AliOS Things开发（C），选择硬件类型haas100，选择解决方案WI-FI设备连接阿里云示例，导入代码

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672314870208-e6d8e277-e910-4fe4-954e-52ac186e1d10.png#averageHue=%23856a43&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=489&id=u447ba661&name=image.png&originHeight=978&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=206807&status=done&style=none&taskId=u72f3e90d-5a8d-4006-8215-ba4ad61e89f&title=&width=959)

2. 在阿里云物联网平台上创建一个自定义品类的产品，并添加属性、事件、服务三个自定义功能，添加完成后将产品发布上线，如下图所示

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672315638599-4d5bca45-e451-4ff2-9efa-3570658df315.png#averageHue=%23fbfaf9&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=459&id=u4d2413b6&name=image.png&originHeight=918&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=198334&status=done&style=none&taskId=u27b55b84-18b0-4a88-8180-b04d09fb5e0&title=&width=959)

3. 发布上线产品后，选择添加设备，添加完设备后，即可得到用于设备进行连接的三元组，我们将在后面的程序中用到它

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672316014914-a75c5d1d-aa83-4843-b979-eb62d69f097d.png#averageHue=%23fcfbf9&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=459&id=u161bb086&name=image.png&originHeight=918&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=202483&status=done&style=none&taskId=u31a869be-1fdf-40b6-8843-d593d87a4bc&title=&width=959)

4. 将`solutions/wifi_demo/seperate_app/data_model_basic_demo.c`主程序的改为如下代码
```c
/*
 * 这个例程演示了用SDK配置MQTT参数并建立连接, 之后创建2个线程
 *
 * + 一个线程用于保活长连接
 * + 一个线程用于接收消息, 并在有消息到达时进入默认的数据回调, 在连接状态变化时进入事件回调
 *
 * 接着演示了在MQTT连接上进行属性上报, 事件上报, 以及处理收到的属性设置, 服务调用, 取消这些代码段落的注释即可观察运行效果
 *
 * 需要用户关注或修改的部分, 已经用 TODO 在注释中标明
 *
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <aos/kernel.h>

#include "aiot_state_api.h"
#include "aiot_sysdep_api.h"
#include "aiot_mqtt_api.h"
#include "aiot_dm_api.h"

/* 位于portfiles/aiot_port文件夹下的系统适配函数集合 */
extern aiot_sysdep_portfile_t g_aiot_sysdep_portfile;

/* 位于external/ali_ca_cert.c中的服务器证书 */
extern const char *ali_ca_cert;


static uint8_t g_mqtt_process_thread_running = 0;
static uint8_t g_mqtt_recv_thread_running = 0;

/* TODO: 如果要关闭日志, 就把这个函数实现为空, 如果要减少日志, 可根据code选择不打印
 *
 * 例如: [1577589489.033][LK-0317] mqtt_basic_demo&a13FN5TplKq
 *
 * 上面这条日志的code就是0317(十六进制), code值的定义见core/aiot_state_api.h
 *
 */

/* 日志回调函数, SDK的日志会从这里输出 */
int32_t demo_state_logcb(int32_t code, char *message)
{
    printf("%s", message);
    return 0;
}

/* MQTT事件回调函数, 当网络连接/重连/断开时被触发, 事件定义见core/aiot_mqtt_api.h */
void demo_mqtt_event_handler(void *handle, const aiot_mqtt_event_t *event, void *userdata)
{
    switch (event->type) {
        /* SDK因为用户调用了aiot_mqtt_connect()接口, 与mqtt服务器建立连接已成功 */
        case AIOT_MQTTEVT_CONNECT: {
            printf("AIOT_MQTTEVT_CONNECT\n");
        }
        break;

        /* SDK因为网络状况被动断连后, 自动发起重连已成功 */
        case AIOT_MQTTEVT_RECONNECT: {
            printf("AIOT_MQTTEVT_RECONNECT\n");
        }
        break;

        /* SDK因为网络的状况而被动断开了连接, network是底层读写失败, heartbeat是没有按预期得到服务端心跳应答 */
        case AIOT_MQTTEVT_DISCONNECT: {
            char *cause = (event->data.disconnect == AIOT_MQTTDISCONNEVT_NETWORK_DISCONNECT) ? ("network disconnect") :
                          ("heartbeat disconnect");
            printf("AIOT_MQTTEVT_DISCONNECT: %s\n", cause);
        }
        break;

        default: {

        }
    }
}

/* 执行aiot_mqtt_process的线程, 包含心跳发送和QoS1消息重发 */
void *demo_mqtt_process_thread(void *args)
{
    int32_t res = STATE_SUCCESS;

    while (g_mqtt_process_thread_running) {
        res = aiot_mqtt_process(args);
        if (res == STATE_USER_INPUT_EXEC_DISABLED) {
            break;
        }
        aos_msleep(1000);
    }
    return NULL;
}

/* 执行aiot_mqtt_recv的线程, 包含网络自动重连和从服务器收取MQTT消息 */
void *demo_mqtt_recv_thread(void *args)
{
    int32_t res = STATE_SUCCESS;

    while (g_mqtt_recv_thread_running) {
        res = aiot_mqtt_recv(args);
        if (res < STATE_SUCCESS) {
            if (res == STATE_USER_INPUT_EXEC_DISABLED) {
                break;
            }
            aos_msleep(1000);
        }
    }
    return NULL;
}

/* 用户数据接收处理回调函数 */
static void demo_dm_recv_handler(void *dm_handle, const aiot_dm_recv_t *recv, void *userdata)
{
    printf("demo_dm_recv_handler, type = %d\r\n", recv->type);

    switch (recv->type) {

        /* 属性上报, 事件上报, 获取期望属性值或者删除期望属性值的应答 */
        case AIOT_DMRECV_GENERIC_REPLY: {
            printf("msg_id = %d, code = %d, data = %.*s, message = %.*s\r\n",
                   recv->data.generic_reply.msg_id,
                   recv->data.generic_reply.code,
                   recv->data.generic_reply.data_len,
                   recv->data.generic_reply.data,
                   recv->data.generic_reply.message_len,
                   recv->data.generic_reply.message);
        }
        break;

        /* 属性设置 */
        case AIOT_DMRECV_PROPERTY_SET: {
            printf("msg_id = %ld, params = %.*s\r\n",
                   (unsigned long)recv->data.property_set.msg_id,
                   recv->data.property_set.params_len,
                   recv->data.property_set.params);

            /* TODO: 以下代码演示如何对来自云平台的属性设置指令进行应答, 用户可取消注释查看演示效果 */
            /*
            {
                aiot_dm_msg_t msg;

                memset(&msg, 0, sizeof(aiot_dm_msg_t));
                msg.type = AIOT_DMMSG_PROPERTY_SET_REPLY;
                msg.data.property_set_reply.msg_id = recv->data.property_set.msg_id;
                msg.data.property_set_reply.code = 200;
                msg.data.property_set_reply.data = "{}";
                int32_t res = aiot_dm_send(dm_handle, &msg);
                if (res < 0) {
                    printf("aiot_dm_send failed\r\n");
                }
            }
            */
        }
        break;

        /* 异步服务调用 */
        case AIOT_DMRECV_ASYNC_SERVICE_INVOKE: {
            printf("msg_id = %ld, service_id = %s, params = %.*s\r\n",
                   (unsigned long)recv->data.async_service_invoke.msg_id,
                   recv->data.async_service_invoke.service_id,
                   recv->data.async_service_invoke.params_len,
                   recv->data.async_service_invoke.params);

            /* TODO: 以下代码演示如何对来自云平台的异步服务调用进行应答, 用户可取消注释查看演示效果
             *
             * 注意: 如果用户在回调函数外进行应答, 需要自行保存msg_id, 因为回调函数入参在退出回调函数后将被SDK销毁, 不可以再访问到
             */

            
            {
                aiot_dm_msg_t msg;

                memset(&msg, 0, sizeof(aiot_dm_msg_t));
                msg.type = AIOT_DMMSG_ASYNC_SERVICE_REPLY;
                msg.data.async_service_reply.msg_id = recv->data.async_service_invoke.msg_id;
                msg.data.async_service_reply.code = 200;
                msg.data.async_service_reply.service_id = "ToggleLightSwitch";
                msg.data.async_service_reply.data = "{\"dataA\": 20}";
                int32_t res = aiot_dm_send(dm_handle, &msg);
                if (res < 0) {
                    printf("aiot_dm_send failed\r\n");
                }else{
                    printf("aiot_dm_send success\r\n");
                }
            }
            
        }
        break;

        /* 同步服务调用 */
        case AIOT_DMRECV_SYNC_SERVICE_INVOKE: {
            printf("msg_id = %ld, rrpc_id = %s, service_id = %s, params = %.*s\r\n",
                   (unsigned long)recv->data.sync_service_invoke.msg_id,
                   recv->data.sync_service_invoke.rrpc_id,
                   recv->data.sync_service_invoke.service_id,
                   recv->data.sync_service_invoke.params_len,
                   recv->data.sync_service_invoke.params);

            /* TODO: 以下代码演示如何对来自云平台的同步服务调用进行应答, 用户可取消注释查看演示效果
             *
             * 注意: 如果用户在回调函数外进行应答, 需要自行保存msg_id和rrpc_id字符串, 因为回调函数入参在退出回调函数后将被SDK销毁, 不可以再访问到
             */

            /*
            {
                aiot_dm_msg_t msg;

                memset(&msg, 0, sizeof(aiot_dm_msg_t));
                msg.type = AIOT_DMMSG_SYNC_SERVICE_REPLY;
                msg.data.sync_service_reply.rrpc_id = recv->data.sync_service_invoke.rrpc_id;
                msg.data.sync_service_reply.msg_id = recv->data.sync_service_invoke.msg_id;
                msg.data.sync_service_reply.code = 200;
                msg.data.sync_service_reply.service_id = "SetLightSwitchTimer";
                msg.data.sync_service_reply.data = "{}";
                int32_t res = aiot_dm_send(dm_handle, &msg);
                if (res < 0) {
                    printf("aiot_dm_send failed\r\n");
                }
            }
            */
        }
        break;

        /* 下行二进制数据 */
        case AIOT_DMRECV_RAW_DATA: {
            printf("raw data len = %d\r\n", recv->data.raw_data.data_len);
            /* TODO: 以下代码演示如何发送二进制格式数据, 若使用需要有相应的数据透传脚本部署在云端 */
            /*
            {
                aiot_dm_msg_t msg;
                uint8_t raw_data[] = {0x01, 0x02};

                memset(&msg, 0, sizeof(aiot_dm_msg_t));
                msg.type = AIOT_DMMSG_RAW_DATA;
                msg.data.raw_data.data = raw_data;
                msg.data.raw_data.data_len = sizeof(raw_data);
                aiot_dm_send(dm_handle, &msg);
            }
            */
        }
        break;

        /* 二进制格式的同步服务调用, 比单纯的二进制数据消息多了个rrpc_id */
        case AIOT_DMRECV_RAW_SYNC_SERVICE_INVOKE: {
            printf("raw sync service rrpc_id = %s, data_len = %d\r\n",
                   recv->data.raw_service_invoke.rrpc_id,
                   recv->data.raw_service_invoke.data_len);
        }
        break;

        default:
            break;
    }
}

/* 属性上报函数演示 */
int32_t demo_send_property_post(void *dm_handle, char *params)
{
    aiot_dm_msg_t msg;

    memset(&msg, 0, sizeof(aiot_dm_msg_t));
    msg.type = AIOT_DMMSG_PROPERTY_POST;
    msg.data.property_post.params = params;

    return aiot_dm_send(dm_handle, &msg);
}

/* 事件上报函数演示 */
int32_t demo_send_event_post(void *dm_handle, char *event_id, char *params)
{
    aiot_dm_msg_t msg;

    memset(&msg, 0, sizeof(aiot_dm_msg_t));
    msg.type = AIOT_DMMSG_EVENT_POST;
    msg.data.event_post.event_id = event_id;
    msg.data.event_post.params = params;

    return aiot_dm_send(dm_handle, &msg);
}

/* 演示了获取属性LightSwitch的期望值, 用户可将此函数加入到main函数中运行演示 */
int32_t demo_send_get_desred_requset(void *dm_handle)
{
    aiot_dm_msg_t msg;

    memset(&msg, 0, sizeof(aiot_dm_msg_t));
    msg.type = AIOT_DMMSG_GET_DESIRED;
    msg.data.get_desired.params = "[\"LightSwitch\"]";

    return aiot_dm_send(dm_handle, &msg);
}

/* 演示了删除属性LightSwitch的期望值, 用户可将此函数加入到main函数中运行演示 */
int32_t demo_send_delete_desred_requset(void *dm_handle)
{
    aiot_dm_msg_t msg;

    memset(&msg, 0, sizeof(aiot_dm_msg_t));
    msg.type = AIOT_DMMSG_DELETE_DESIRED;
    msg.data.get_desired.params = "{\"LightSwitch\":{}}";

    return aiot_dm_send(dm_handle, &msg);
}


int demo_main(int argc, char *argv[])
{
    int32_t     res = STATE_SUCCESS;
    void       *dm_handle = NULL;
    void       *mqtt_handle = NULL;
    char       *url = "iot-as-mqtt.cn-shanghai.aliyuncs.com"; /* 阿里云平台上海站点的域名后缀 */
    char        host[100] = {0}; /* 用这个数组拼接设备连接的云平台站点全地址, 规则是 ${productKey}.iot-as-mqtt.cn-shanghai.aliyuncs.com */
    uint16_t    port = 443;      /* 无论设备是否使用TLS连接阿里云平台, 目的端口都是443 */
    aiot_sysdep_network_cred_t cred; /* 安全凭据结构体, 如果要用TLS, 这个结构体中配置CA证书等参数 */

    /* TODO: 替换为自己设备的三元组 */
    char *product_key       = "hxxquuiBzXM";
    char *device_name       = "dev1";
    char *device_secret     = "be369f2c9d9a0cb83d04ee6ef8071b31";

    /* 配置SDK的底层依赖 */
    aiot_sysdep_set_portfile(&g_aiot_sysdep_portfile);
    /* 配置SDK的日志输出 */
    aiot_state_set_logcb(demo_state_logcb);

    /* 创建SDK的安全凭据, 用于建立TLS连接 */
    memset(&cred, 0, sizeof(aiot_sysdep_network_cred_t));
    cred.option = AIOT_SYSDEP_NETWORK_CRED_SVRCERT_CA;  /* 使用RSA证书校验MQTT服务端 */
    cred.max_tls_fragment = 16384; /* 最大的分片长度为16K, 其它可选值还有4K, 2K, 1K, 0.5K */
    cred.sni_enabled = 1;                               /* TLS建连时, 支持Server Name Indicator */
    cred.x509_server_cert = ali_ca_cert;                 /* 用来验证MQTT服务端的RSA根证书 */
    cred.x509_server_cert_len = strlen(ali_ca_cert);     /* 用来验证MQTT服务端的RSA根证书长度 */

    /* 创建1个MQTT客户端实例并内部初始化默认参数 */
    mqtt_handle = aiot_mqtt_init();
    if (mqtt_handle == NULL) {
        printf("aiot_mqtt_init failed\n");
        return -1;
    }

    snprintf(host, 100, "%s.%s", product_key, url);
    /* 配置MQTT服务器地址 */
    aiot_mqtt_setopt(mqtt_handle, AIOT_MQTTOPT_HOST, (void *)host);
    /* 配置MQTT服务器端口 */
    aiot_mqtt_setopt(mqtt_handle, AIOT_MQTTOPT_PORT, (void *)&port);
    /* 配置设备productKey */
    aiot_mqtt_setopt(mqtt_handle, AIOT_MQTTOPT_PRODUCT_KEY, (void *)product_key);
    /* 配置设备deviceName */
    aiot_mqtt_setopt(mqtt_handle, AIOT_MQTTOPT_DEVICE_NAME, (void *)device_name);
    /* 配置设备deviceSecret */
    aiot_mqtt_setopt(mqtt_handle, AIOT_MQTTOPT_DEVICE_SECRET, (void *)device_secret);
    /* 配置网络连接的安全凭据, 上面已经创建好了 */
    aiot_mqtt_setopt(mqtt_handle, AIOT_MQTTOPT_NETWORK_CRED, (void *)&cred);
    /* 配置MQTT事件回调函数 */
    aiot_mqtt_setopt(mqtt_handle, AIOT_MQTTOPT_EVENT_HANDLER, (void *)demo_mqtt_event_handler);

    /* 创建DATA-MODEL实例 */
    dm_handle = aiot_dm_init();
    if (dm_handle == NULL) {
        printf("aiot_dm_init failed");
        return -1;
    }
    /* 配置MQTT实例句柄 */
    aiot_dm_setopt(dm_handle, AIOT_DMOPT_MQTT_HANDLE, mqtt_handle);
    /* 配置消息接收处理回调函数 */
    aiot_dm_setopt(dm_handle, AIOT_DMOPT_RECV_HANDLER, (void *)demo_dm_recv_handler);

    /* 与服务器建立MQTT连接 */
    res = aiot_mqtt_connect(mqtt_handle);
    if (res < STATE_SUCCESS) {
        /* 尝试建立连接失败, 销毁MQTT实例, 回收资源 */
        aiot_mqtt_deinit(&mqtt_handle);
        printf("aiot_mqtt_connect failed: -0x%04X\n", -res);
        return -1;
    }

    /* 创建一个单独的线程, 专用于执行aiot_mqtt_process, 它会自动发送心跳保活, 以及重发QoS1的未应答报文 */
    g_mqtt_process_thread_running = 1;
    res = aos_task_new("demo_mqtt_process", demo_mqtt_process_thread, mqtt_handle, 4096);
    // res = pthread_create(&g_mqtt_process_thread, NULL, demo_mqtt_process_thread, mqtt_handle);
    if (res != 0) {
        printf("create demo_mqtt_process_thread failed: %d\n", res);
        return -1;
    }

    /* 创建一个单独的线程用于执行aiot_mqtt_recv, 它会循环收取服务器下发的MQTT消息, 并在断线时自动重连 */
    g_mqtt_recv_thread_running = 1;
    res = aos_task_new("demo_mqtt_process", demo_mqtt_recv_thread, mqtt_handle, 4096);
    // res = pthread_create(&g_mqtt_recv_thread, NULL, demo_mqtt_recv_thread, mqtt_handle);
    if (res != 0) {
        printf("create demo_mqtt_recv_thread failed: %d\n", res);
        return -1;
    }
    
    char msg[20];
    uint16_t count = 0;
    /* 主循环进入休眠 */
    while (1) {
        /* TODO: 以下代码演示了简单的属性上报和事件上报, 用户可取消注释观察演示效果 */
        count++;
        snprintf(msg,20,"{\"prop1\": %d}",count);
        demo_send_property_post(dm_handle, msg);
        // demo_send_property_post(dm_handle, "{\"prop1\": 0}");
        demo_send_event_post(dm_handle, "event1", "{\"ErrorCode\": 0}");

        aos_msleep(10000);
    }

    /* 断开MQTT连接, 一般不会运行到这里 */
    res = aiot_mqtt_disconnect(mqtt_handle);
    if (res < STATE_SUCCESS) {
        aiot_mqtt_deinit(&mqtt_handle);
        printf("aiot_mqtt_disconnect failed: -0x%04X\n", -res);
        return -1;
    }

    /* 销毁DATA-MODEL实例, 一般不会运行到这里 */
    res = aiot_dm_deinit(&dm_handle);
    if (res < STATE_SUCCESS) {
        printf("aiot_dm_deinit failed: -0x%04X\n", -res);
        return -1;
    }

    /* 销毁MQTT实例, 一般不会运行到这里 */
    res = aiot_mqtt_deinit(&mqtt_handle);
    if (res < STATE_SUCCESS) {
        printf("aiot_mqtt_deinit failed: -0x%04X\n", -res);
        return -1;
    }

    g_mqtt_process_thread_running = 0;
    g_mqtt_recv_thread_running = 0;

    return 0;
}


```

5. 编译并烧录程序，打开串口，输入配网指令`netmgr -t wifi -c ssid password`进行联网，网络连接正常且程序运行无误后，终端输出`success`，阿里云物联网平台上设备也显示在线

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672316339965-9ae76063-cfb6-4c8d-b20d-b529bc9f6bc0.png#averageHue=%232a2f38&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=488&id=ufe43e350&name=image.png&originHeight=976&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=344977&status=done&style=none&taskId=ubd3ff2ec-fddf-4201-a469-26106bc873a&title=&width=959)
## 4. 实验结果

1. 属性上报结果：打开`阿里云物联网平台-->设备管理-->设备-->物模型数据-->运行状态`即可看到属性上报结果

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672316588457-f89108fe-a214-4f90-b501-4b7cbdf2a7af.png#averageHue=%23fcfbf9&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=459&id=u9842c1fa&name=image.png&originHeight=918&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=157826&status=done&style=none&taskId=u284fba84-5e24-47b5-92fd-3693e874a4d&title=&width=959)

2. 事件上报结果：打开`阿里云物联网平台-->设备管理-->设备-->物模型数据-->事件管理`即可看到事件上报结果

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672316681926-46e3da02-cbd2-40a0-8cf3-a3d87fbf6800.png#averageHue=%23fbfaf8&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=459&id=u45f6a651&name=image.png&originHeight=918&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=192427&status=done&style=none&taskId=uce605bc3-96e1-41ab-b662-6456d743d83&title=&width=959)

3. 服务调用结果：打开`阿里云物联网平台-->监控运维-->在线调试-->服务调用`，点击发送指令，发送一条空指令，即可在右边实时日志中看到服务调用结果，也可在`设备管理-->设备-->物模型数据-->服务调用`中看到服务调用结果

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672316921299-97d1cef2-bf7a-4140-926b-73a830c6808f.png#averageHue=%23f6f5f4&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=459&id=u6b47cfc2&name=image.png&originHeight=918&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=333156&status=done&style=none&taskId=u86a04c4c-a15f-46de-bb66-af6f028fb6b&title=&width=959)
![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672316960257-d48a0a0b-548f-4051-97de-aa8c2fc72ad6.png#averageHue=%23fbfaf8&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=459&id=ud5726035&name=image.png&originHeight=918&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=199635&status=done&style=none&taskId=ua1093d2b-63ef-4992-a29b-4762532e712&title=&width=959)
# 作业三、树莓派网关产品
## 0. 网关产品的作用

- 网关是一种充当转换重任的计算机系统或设备，在使用不同的通信协议，数据格式或语言,甚至体系结构完全不同的两种系统时，网关是一个翻译器。与网桥只是简单地传送信息不同，网关对收到的信息要重新打包，以适应目的系统的需求。同时，网关也可以提供过滤和安全功能。
- 简单来说，网关是设备与路由器之间的桥梁，正确的网关配置才能保证用户可以正常上网。
## 1. 实验要求

- 在阿里云边缘计算平台上创建一个网关产品
- 使用树莓派作为边缘计算的网关
## 2. 实验环境

- 硬件平台：Raspberry Pi 4B（8GB RAM）
- 操作系统：Raspberry Pi OS（64-bit）
- 开发平台：VSCode Remote-SSH extension
- 阿里云边缘计算平台，网址：[link](https://iot.console.aliyun.com/le/instance/list)   
## 3. 实验过程

1. 将树莓派和主机连入同一个wifi中，在VSCode中打开SSH进行连接

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672319102534-5872fa59-02ba-4a80-a5ea-2ce2e187814b.png#averageHue=%23262b32&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=488&id=ucbc662fb&name=image.png&originHeight=975&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=66790&status=done&style=none&taskId=u7027d718-8052-4365-853e-c46ffaed253&title=&width=959)

2. 打开阿里云边缘计算平台，点击创建边缘实例，再点击新建网关产品创建网关产品，接着点击新建网关设备创建网关设备，最后设置完成后点击确定

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672319919060-acbec924-e3b7-4633-b6e3-7d1a21202191.png#averageHue=%23fcfbfb&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=243&id=u74027e9e&name=image.png&originHeight=504&originWidth=534&originalType=binary&ratio=1&rotation=0&showTitle=false&size=21597&status=done&style=none&taskId=u48e1ae6c-04d9-4d6a-9697-800fb3d5603&title=&width=257)![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672320057027-4b48be15-33f0-46ec-8f1d-9335a10afc98.png#averageHue=%23fcfcfc&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=211&id=u7faab699&name=image.png&originHeight=457&originWidth=521&originalType=binary&ratio=1&rotation=0&showTitle=false&size=19748&status=done&style=none&taskId=u4c32f817-f57d-43d3-a833-37661baddab&title=&width=240)![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672320085508-77a1aeb4-75ad-4c03-8820-d3bec570f794.png#averageHue=%23fcfbfb&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=272&id=u272a2d06&name=image.png&originHeight=605&originWidth=534&originalType=binary&ratio=1&rotation=0&showTitle=false&size=27277&status=done&style=none&taskId=uec32162c-f552-42ff-8f57-7e1acd85d48&title=&width=240)

3. 创建完成后点击实例右侧的软件安装，选择硬件平台对应的软件版本，生成相应的安装代码，复制安装代码到树莓派终端进行安装，等待安装完成

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672320782165-0a7c5ccd-cec3-40ba-b63e-b8123b04b81d.png#averageHue=%23282c34&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=488&id=u81d7da99&name=image.png&originHeight=975&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=132218&status=done&style=none&taskId=u4ebb0911-8c2c-44ef-b844-6d082367151&title=&width=959)
## 4. 实验结果

1. 在树梅派终端中输入指令`sudo ./link-iot-edge-standard.sh --status`查看Link IoT Edge核心服务的运行状态，全为`active`则正常运行

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672321083028-0581e87d-25da-4a86-b642-a3ada2449764.png#averageHue=%23282c34&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=488&id=u397cea9b&name=image.png&originHeight=975&originWidth=1918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=208811&status=done&style=none&taskId=u7060e175-5255-45a8-aae2-5f807d7940e&title=&width=959)

2. 也可以在边缘计算控制台，左侧导航栏选择边缘实例，在已创建好的边缘实例右侧单击查看进入实例详情页面，选择网关，查看网关状态，状态显示为在线则运行正常

![image.png](https://cdn.nlark.com/yuque/0/2022/png/34751784/1672321180968-8b82e593-ef7e-48aa-bcbe-6a3d596622ca.png#averageHue=%23fbfbfa&clientId=u39f68c61-ff9b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=130&id=u6172dc8d&name=image.png&originHeight=259&originWidth=1537&originalType=binary&ratio=1&rotation=0&showTitle=false&size=31972&status=done&style=none&taskId=u9076ee3b-6105-4308-9876-2e499d27e4d&title=&width=769)
