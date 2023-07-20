package com.osx.broker.consumer;

import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.broker.ServiceContainer;
import com.osx.broker.message.MessageExt;
import com.osx.core.config.MetaInfo;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.utils.JsonUtil;
import io.grpc.stub.StreamObserver;

/**
 * 放在源头，用于接听远端返回
 */
public class SourceGrpcEventHandler extends GrpcEventHandler{

    com.google.protobuf.Parser  parser;
    StreamObserver respStreamObserver;


    public  SourceGrpcEventHandler(StreamObserver  respStreamObserver,
                                   com.google.protobuf.Parser  parser){
        super(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        this.parser=parser;
        this.respStreamObserver = respStreamObserver;
    }

    @Override
    protected void handleMessage(MessageExt message) {

        try {
           Object data =  parser.parseFrom(message.getBody());
            respStreamObserver.onNext(data);
        } catch (InvalidProtocolBufferException e) {
            logger.error("");
        }
    }

    @Override
    protected void handleError(MessageExt message) {
        try {
            ExceptionInfo exceptionInfo = JsonUtil.json2Object(message.getBody(), ExceptionInfo.class);
            respStreamObserver.onError(new Throwable(exceptionInfo.getMessage()));
        }finally {
            String  topic =message.getTopic();
            ServiceContainer.transferQueueManager.onCompleted(topic);
        }

    }

    @Override
    protected void handleComplete(MessageExt message) {
        try {
            respStreamObserver.onCompleted();
        }finally {
            String  topic =message.getTopic();
            ServiceContainer.transferQueueManager.onCompleted(topic);
        }

    }

    @Override
    protected void handleInit(MessageExt message) {

    }
}