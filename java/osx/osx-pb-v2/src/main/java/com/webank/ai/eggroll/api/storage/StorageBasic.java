// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: eggroll/storage-basic.proto

package com.webank.ai.eggroll.api.storage;

public final class StorageBasic {
  private StorageBasic() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  /**
   * <pre>
   * todo: merge with Stores class
   * </pre>
   *
   * Protobuf enum {@code com.webank.ai.eggroll.api.storage.StorageType}
   */
  public enum StorageType
      implements com.google.protobuf.ProtocolMessageEnum {
    /**
     * <code>LEVEL_DB = 0;</code>
     */
    LEVEL_DB(0),
    /**
     * <code>IN_MEMORY = 1;</code>
     */
    IN_MEMORY(1),
    /**
     * <code>LMDB = 2;</code>
     */
    LMDB(2),
    /**
     * <code>REDIS = 3;</code>
     */
    REDIS(3),
    UNRECOGNIZED(-1),
    ;

    /**
     * <code>LEVEL_DB = 0;</code>
     */
    public static final int LEVEL_DB_VALUE = 0;
    /**
     * <code>IN_MEMORY = 1;</code>
     */
    public static final int IN_MEMORY_VALUE = 1;
    /**
     * <code>LMDB = 2;</code>
     */
    public static final int LMDB_VALUE = 2;
    /**
     * <code>REDIS = 3;</code>
     */
    public static final int REDIS_VALUE = 3;


    public final int getNumber() {
      if (this == UNRECOGNIZED) {
        throw new java.lang.IllegalArgumentException(
            "Can't get the number of an unknown enum value.");
      }
      return value;
    }

    /**
     * @param value The numeric wire value of the corresponding enum entry.
     * @return The enum associated with the given numeric wire value.
     * @deprecated Use {@link #forNumber(int)} instead.
     */
    @java.lang.Deprecated
    public static StorageType valueOf(int value) {
      return forNumber(value);
    }

    /**
     * @param value The numeric wire value of the corresponding enum entry.
     * @return The enum associated with the given numeric wire value.
     */
    public static StorageType forNumber(int value) {
      switch (value) {
        case 0: return LEVEL_DB;
        case 1: return IN_MEMORY;
        case 2: return LMDB;
        case 3: return REDIS;
        default: return null;
      }
    }

    public static com.google.protobuf.Internal.EnumLiteMap<StorageType>
        internalGetValueMap() {
      return internalValueMap;
    }
    private static final com.google.protobuf.Internal.EnumLiteMap<
        StorageType> internalValueMap =
          new com.google.protobuf.Internal.EnumLiteMap<StorageType>() {
            public StorageType findValueByNumber(int number) {
              return StorageType.forNumber(number);
            }
          };

    public final com.google.protobuf.Descriptors.EnumValueDescriptor
        getValueDescriptor() {
      if (this == UNRECOGNIZED) {
        throw new java.lang.IllegalStateException(
            "Can't get the descriptor of an unrecognized enum value.");
      }
      return getDescriptor().getValues().get(ordinal());
    }
    public final com.google.protobuf.Descriptors.EnumDescriptor
        getDescriptorForType() {
      return getDescriptor();
    }
    public static final com.google.protobuf.Descriptors.EnumDescriptor
        getDescriptor() {
      return com.webank.ai.eggroll.api.storage.StorageBasic.getDescriptor().getEnumTypes().get(0);
    }

    private static final StorageType[] VALUES = values();

    public static StorageType valueOf(
        com.google.protobuf.Descriptors.EnumValueDescriptor desc) {
      if (desc.getType() != getDescriptor()) {
        throw new java.lang.IllegalArgumentException(
          "EnumValueDescriptor is not for this type.");
      }
      if (desc.getIndex() == -1) {
        return UNRECOGNIZED;
      }
      return VALUES[desc.getIndex()];
    }

    private final int value;

    private StorageType(int value) {
      this.value = value;
    }

    // @@protoc_insertion_point(enum_scope:com.webank.ai.eggroll.api.storage.StorageType)
  }

  public interface StorageLocatorOrBuilder extends
      // @@protoc_insertion_point(interface_extends:com.webank.ai.eggroll.api.storage.StorageLocator)
      com.google.protobuf.MessageOrBuilder {

    /**
     * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
     * @return The enum numeric value on the wire for type.
     */
    int getTypeValue();
    /**
     * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
     * @return The type.
     */
    com.webank.ai.eggroll.api.storage.StorageBasic.StorageType getType();

    /**
     * <code>string namespace = 2;</code>
     * @return The namespace.
     */
    java.lang.String getNamespace();
    /**
     * <code>string namespace = 2;</code>
     * @return The bytes for namespace.
     */
    com.google.protobuf.ByteString
        getNamespaceBytes();

    /**
     * <code>string name = 3;</code>
     * @return The name.
     */
    java.lang.String getName();
    /**
     * <code>string name = 3;</code>
     * @return The bytes for name.
     */
    com.google.protobuf.ByteString
        getNameBytes();

    /**
     * <code>int32 fragment = 4;</code>
     * @return The fragment.
     */
    int getFragment();
  }
  /**
   * <pre>
   * information of storage
   * todo: merge with StoreInfo class
   * </pre>
   *
   * Protobuf type {@code com.webank.ai.eggroll.api.storage.StorageLocator}
   */
  public static final class StorageLocator extends
      com.google.protobuf.GeneratedMessageV3 implements
      // @@protoc_insertion_point(message_implements:com.webank.ai.eggroll.api.storage.StorageLocator)
      StorageLocatorOrBuilder {
  private static final long serialVersionUID = 0L;
    // Use StorageLocator.newBuilder() to construct.
    private StorageLocator(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
      super(builder);
    }
    private StorageLocator() {
      type_ = 0;
      namespace_ = "";
      name_ = "";
    }

    @java.lang.Override
    @SuppressWarnings({"unused"})
    protected java.lang.Object newInstance(
        UnusedPrivateParameter unused) {
      return new StorageLocator();
    }

    @java.lang.Override
    public final com.google.protobuf.UnknownFieldSet
    getUnknownFields() {
      return this.unknownFields;
    }
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return com.webank.ai.eggroll.api.storage.StorageBasic.internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return com.webank.ai.eggroll.api.storage.StorageBasic.internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.Builder.class);
    }

    public static final int TYPE_FIELD_NUMBER = 1;
    private int type_ = 0;
    /**
     * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
     * @return The enum numeric value on the wire for type.
     */
    @java.lang.Override public int getTypeValue() {
      return type_;
    }
    /**
     * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
     * @return The type.
     */
    @java.lang.Override public com.webank.ai.eggroll.api.storage.StorageBasic.StorageType getType() {
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageType result = com.webank.ai.eggroll.api.storage.StorageBasic.StorageType.forNumber(type_);
      return result == null ? com.webank.ai.eggroll.api.storage.StorageBasic.StorageType.UNRECOGNIZED : result;
    }

    public static final int NAMESPACE_FIELD_NUMBER = 2;
    @SuppressWarnings("serial")
    private volatile java.lang.Object namespace_ = "";
    /**
     * <code>string namespace = 2;</code>
     * @return The namespace.
     */
    @java.lang.Override
    public java.lang.String getNamespace() {
      java.lang.Object ref = namespace_;
      if (ref instanceof java.lang.String) {
        return (java.lang.String) ref;
      } else {
        com.google.protobuf.ByteString bs = 
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        namespace_ = s;
        return s;
      }
    }
    /**
     * <code>string namespace = 2;</code>
     * @return The bytes for namespace.
     */
    @java.lang.Override
    public com.google.protobuf.ByteString
        getNamespaceBytes() {
      java.lang.Object ref = namespace_;
      if (ref instanceof java.lang.String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        namespace_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }

    public static final int NAME_FIELD_NUMBER = 3;
    @SuppressWarnings("serial")
    private volatile java.lang.Object name_ = "";
    /**
     * <code>string name = 3;</code>
     * @return The name.
     */
    @java.lang.Override
    public java.lang.String getName() {
      java.lang.Object ref = name_;
      if (ref instanceof java.lang.String) {
        return (java.lang.String) ref;
      } else {
        com.google.protobuf.ByteString bs = 
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        name_ = s;
        return s;
      }
    }
    /**
     * <code>string name = 3;</code>
     * @return The bytes for name.
     */
    @java.lang.Override
    public com.google.protobuf.ByteString
        getNameBytes() {
      java.lang.Object ref = name_;
      if (ref instanceof java.lang.String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        name_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }

    public static final int FRAGMENT_FIELD_NUMBER = 4;
    private int fragment_ = 0;
    /**
     * <code>int32 fragment = 4;</code>
     * @return The fragment.
     */
    @java.lang.Override
    public int getFragment() {
      return fragment_;
    }

    private byte memoizedIsInitialized = -1;
    @java.lang.Override
    public final boolean isInitialized() {
      byte isInitialized = memoizedIsInitialized;
      if (isInitialized == 1) return true;
      if (isInitialized == 0) return false;

      memoizedIsInitialized = 1;
      return true;
    }

    @java.lang.Override
    public void writeTo(com.google.protobuf.CodedOutputStream output)
                        throws java.io.IOException {
      if (type_ != com.webank.ai.eggroll.api.storage.StorageBasic.StorageType.LEVEL_DB.getNumber()) {
        output.writeEnum(1, type_);
      }
      if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(namespace_)) {
        com.google.protobuf.GeneratedMessageV3.writeString(output, 2, namespace_);
      }
      if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(name_)) {
        com.google.protobuf.GeneratedMessageV3.writeString(output, 3, name_);
      }
      if (fragment_ != 0) {
        output.writeInt32(4, fragment_);
      }
      getUnknownFields().writeTo(output);
    }

    @java.lang.Override
    public int getSerializedSize() {
      int size = memoizedSize;
      if (size != -1) return size;

      size = 0;
      if (type_ != com.webank.ai.eggroll.api.storage.StorageBasic.StorageType.LEVEL_DB.getNumber()) {
        size += com.google.protobuf.CodedOutputStream
          .computeEnumSize(1, type_);
      }
      if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(namespace_)) {
        size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, namespace_);
      }
      if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(name_)) {
        size += com.google.protobuf.GeneratedMessageV3.computeStringSize(3, name_);
      }
      if (fragment_ != 0) {
        size += com.google.protobuf.CodedOutputStream
          .computeInt32Size(4, fragment_);
      }
      size += getUnknownFields().getSerializedSize();
      memoizedSize = size;
      return size;
    }

    @java.lang.Override
    public boolean equals(final java.lang.Object obj) {
      if (obj == this) {
       return true;
      }
      if (!(obj instanceof com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator)) {
        return super.equals(obj);
      }
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator other = (com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator) obj;

      if (type_ != other.type_) return false;
      if (!getNamespace()
          .equals(other.getNamespace())) return false;
      if (!getName()
          .equals(other.getName())) return false;
      if (getFragment()
          != other.getFragment()) return false;
      if (!getUnknownFields().equals(other.getUnknownFields())) return false;
      return true;
    }

    @java.lang.Override
    public int hashCode() {
      if (memoizedHashCode != 0) {
        return memoizedHashCode;
      }
      int hash = 41;
      hash = (19 * hash) + getDescriptor().hashCode();
      hash = (37 * hash) + TYPE_FIELD_NUMBER;
      hash = (53 * hash) + type_;
      hash = (37 * hash) + NAMESPACE_FIELD_NUMBER;
      hash = (53 * hash) + getNamespace().hashCode();
      hash = (37 * hash) + NAME_FIELD_NUMBER;
      hash = (53 * hash) + getName().hashCode();
      hash = (37 * hash) + FRAGMENT_FIELD_NUMBER;
      hash = (53 * hash) + getFragment();
      hash = (29 * hash) + getUnknownFields().hashCode();
      memoizedHashCode = hash;
      return hash;
    }

    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(
        java.nio.ByteBuffer data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(
        java.nio.ByteBuffer data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(
        com.google.protobuf.ByteString data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(
        com.google.protobuf.ByteString data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(byte[] data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(
        byte[] data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input, extensionRegistry);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseDelimitedFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseDelimitedFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(
        com.google.protobuf.CodedInputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator parseFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input, extensionRegistry);
    }

    @java.lang.Override
    public Builder newBuilderForType() { return newBuilder(); }
    public static Builder newBuilder() {
      return DEFAULT_INSTANCE.toBuilder();
    }
    public static Builder newBuilder(com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator prototype) {
      return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
    }
    @java.lang.Override
    public Builder toBuilder() {
      return this == DEFAULT_INSTANCE
          ? new Builder() : new Builder().mergeFrom(this);
    }

    @java.lang.Override
    protected Builder newBuilderForType(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      Builder builder = new Builder(parent);
      return builder;
    }
    /**
     * <pre>
     * information of storage
     * todo: merge with StoreInfo class
     * </pre>
     *
     * Protobuf type {@code com.webank.ai.eggroll.api.storage.StorageLocator}
     */
    public static final class Builder extends
        com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
        // @@protoc_insertion_point(builder_implements:com.webank.ai.eggroll.api.storage.StorageLocator)
        com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocatorOrBuilder {
      public static final com.google.protobuf.Descriptors.Descriptor
          getDescriptor() {
        return com.webank.ai.eggroll.api.storage.StorageBasic.internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_descriptor;
      }

      @java.lang.Override
      protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
          internalGetFieldAccessorTable() {
        return com.webank.ai.eggroll.api.storage.StorageBasic.internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_fieldAccessorTable
            .ensureFieldAccessorsInitialized(
                com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.Builder.class);
      }

      // Construct using com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.newBuilder()
      private Builder() {

      }

      private Builder(
          com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
        super(parent);

      }
      @java.lang.Override
      public Builder clear() {
        super.clear();
        bitField0_ = 0;
        type_ = 0;
        namespace_ = "";
        name_ = "";
        fragment_ = 0;
        return this;
      }

      @java.lang.Override
      public com.google.protobuf.Descriptors.Descriptor
          getDescriptorForType() {
        return com.webank.ai.eggroll.api.storage.StorageBasic.internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_descriptor;
      }

      @java.lang.Override
      public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator getDefaultInstanceForType() {
        return com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance();
      }

      @java.lang.Override
      public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator build() {
        com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator result = buildPartial();
        if (!result.isInitialized()) {
          throw newUninitializedMessageException(result);
        }
        return result;
      }

      @java.lang.Override
      public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator buildPartial() {
        com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator result = new com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator(this);
        if (bitField0_ != 0) { buildPartial0(result); }
        onBuilt();
        return result;
      }

      private void buildPartial0(com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator result) {
        int from_bitField0_ = bitField0_;
        if (((from_bitField0_ & 0x00000001) != 0)) {
          result.type_ = type_;
        }
        if (((from_bitField0_ & 0x00000002) != 0)) {
          result.namespace_ = namespace_;
        }
        if (((from_bitField0_ & 0x00000004) != 0)) {
          result.name_ = name_;
        }
        if (((from_bitField0_ & 0x00000008) != 0)) {
          result.fragment_ = fragment_;
        }
      }

      @java.lang.Override
      public Builder clone() {
        return super.clone();
      }
      @java.lang.Override
      public Builder setField(
          com.google.protobuf.Descriptors.FieldDescriptor field,
          java.lang.Object value) {
        return super.setField(field, value);
      }
      @java.lang.Override
      public Builder clearField(
          com.google.protobuf.Descriptors.FieldDescriptor field) {
        return super.clearField(field);
      }
      @java.lang.Override
      public Builder clearOneof(
          com.google.protobuf.Descriptors.OneofDescriptor oneof) {
        return super.clearOneof(oneof);
      }
      @java.lang.Override
      public Builder setRepeatedField(
          com.google.protobuf.Descriptors.FieldDescriptor field,
          int index, java.lang.Object value) {
        return super.setRepeatedField(field, index, value);
      }
      @java.lang.Override
      public Builder addRepeatedField(
          com.google.protobuf.Descriptors.FieldDescriptor field,
          java.lang.Object value) {
        return super.addRepeatedField(field, value);
      }
      @java.lang.Override
      public Builder mergeFrom(com.google.protobuf.Message other) {
        if (other instanceof com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator) {
          return mergeFrom((com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator)other);
        } else {
          super.mergeFrom(other);
          return this;
        }
      }

      public Builder mergeFrom(com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator other) {
        if (other == com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()) return this;
        if (other.type_ != 0) {
          setTypeValue(other.getTypeValue());
        }
        if (!other.getNamespace().isEmpty()) {
          namespace_ = other.namespace_;
          bitField0_ |= 0x00000002;
          onChanged();
        }
        if (!other.getName().isEmpty()) {
          name_ = other.name_;
          bitField0_ |= 0x00000004;
          onChanged();
        }
        if (other.getFragment() != 0) {
          setFragment(other.getFragment());
        }
        this.mergeUnknownFields(other.getUnknownFields());
        onChanged();
        return this;
      }

      @java.lang.Override
      public final boolean isInitialized() {
        return true;
      }

      @java.lang.Override
      public Builder mergeFrom(
          com.google.protobuf.CodedInputStream input,
          com.google.protobuf.ExtensionRegistryLite extensionRegistry)
          throws java.io.IOException {
        if (extensionRegistry == null) {
          throw new java.lang.NullPointerException();
        }
        try {
          boolean done = false;
          while (!done) {
            int tag = input.readTag();
            switch (tag) {
              case 0:
                done = true;
                break;
              case 8: {
                type_ = input.readEnum();
                bitField0_ |= 0x00000001;
                break;
              } // case 8
              case 18: {
                namespace_ = input.readStringRequireUtf8();
                bitField0_ |= 0x00000002;
                break;
              } // case 18
              case 26: {
                name_ = input.readStringRequireUtf8();
                bitField0_ |= 0x00000004;
                break;
              } // case 26
              case 32: {
                fragment_ = input.readInt32();
                bitField0_ |= 0x00000008;
                break;
              } // case 32
              default: {
                if (!super.parseUnknownField(input, extensionRegistry, tag)) {
                  done = true; // was an endgroup tag
                }
                break;
              } // default:
            } // switch (tag)
          } // while (!done)
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
          throw e.unwrapIOException();
        } finally {
          onChanged();
        } // finally
        return this;
      }
      private int bitField0_;

      private int type_ = 0;
      /**
       * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
       * @return The enum numeric value on the wire for type.
       */
      @java.lang.Override public int getTypeValue() {
        return type_;
      }
      /**
       * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
       * @param value The enum numeric value on the wire for type to set.
       * @return This builder for chaining.
       */
      public Builder setTypeValue(int value) {
        type_ = value;
        bitField0_ |= 0x00000001;
        onChanged();
        return this;
      }
      /**
       * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
       * @return The type.
       */
      @java.lang.Override
      public com.webank.ai.eggroll.api.storage.StorageBasic.StorageType getType() {
        com.webank.ai.eggroll.api.storage.StorageBasic.StorageType result = com.webank.ai.eggroll.api.storage.StorageBasic.StorageType.forNumber(type_);
        return result == null ? com.webank.ai.eggroll.api.storage.StorageBasic.StorageType.UNRECOGNIZED : result;
      }
      /**
       * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
       * @param value The type to set.
       * @return This builder for chaining.
       */
      public Builder setType(com.webank.ai.eggroll.api.storage.StorageBasic.StorageType value) {
        if (value == null) {
          throw new NullPointerException();
        }
        bitField0_ |= 0x00000001;
        type_ = value.getNumber();
        onChanged();
        return this;
      }
      /**
       * <code>.com.webank.ai.eggroll.api.storage.StorageType type = 1;</code>
       * @return This builder for chaining.
       */
      public Builder clearType() {
        bitField0_ = (bitField0_ & ~0x00000001);
        type_ = 0;
        onChanged();
        return this;
      }

      private java.lang.Object namespace_ = "";
      /**
       * <code>string namespace = 2;</code>
       * @return The namespace.
       */
      public java.lang.String getNamespace() {
        java.lang.Object ref = namespace_;
        if (!(ref instanceof java.lang.String)) {
          com.google.protobuf.ByteString bs =
              (com.google.protobuf.ByteString) ref;
          java.lang.String s = bs.toStringUtf8();
          namespace_ = s;
          return s;
        } else {
          return (java.lang.String) ref;
        }
      }
      /**
       * <code>string namespace = 2;</code>
       * @return The bytes for namespace.
       */
      public com.google.protobuf.ByteString
          getNamespaceBytes() {
        java.lang.Object ref = namespace_;
        if (ref instanceof String) {
          com.google.protobuf.ByteString b = 
              com.google.protobuf.ByteString.copyFromUtf8(
                  (java.lang.String) ref);
          namespace_ = b;
          return b;
        } else {
          return (com.google.protobuf.ByteString) ref;
        }
      }
      /**
       * <code>string namespace = 2;</code>
       * @param value The namespace to set.
       * @return This builder for chaining.
       */
      public Builder setNamespace(
          java.lang.String value) {
        if (value == null) { throw new NullPointerException(); }
        namespace_ = value;
        bitField0_ |= 0x00000002;
        onChanged();
        return this;
      }
      /**
       * <code>string namespace = 2;</code>
       * @return This builder for chaining.
       */
      public Builder clearNamespace() {
        namespace_ = getDefaultInstance().getNamespace();
        bitField0_ = (bitField0_ & ~0x00000002);
        onChanged();
        return this;
      }
      /**
       * <code>string namespace = 2;</code>
       * @param value The bytes for namespace to set.
       * @return This builder for chaining.
       */
      public Builder setNamespaceBytes(
          com.google.protobuf.ByteString value) {
        if (value == null) { throw new NullPointerException(); }
        checkByteStringIsUtf8(value);
        namespace_ = value;
        bitField0_ |= 0x00000002;
        onChanged();
        return this;
      }

      private java.lang.Object name_ = "";
      /**
       * <code>string name = 3;</code>
       * @return The name.
       */
      public java.lang.String getName() {
        java.lang.Object ref = name_;
        if (!(ref instanceof java.lang.String)) {
          com.google.protobuf.ByteString bs =
              (com.google.protobuf.ByteString) ref;
          java.lang.String s = bs.toStringUtf8();
          name_ = s;
          return s;
        } else {
          return (java.lang.String) ref;
        }
      }
      /**
       * <code>string name = 3;</code>
       * @return The bytes for name.
       */
      public com.google.protobuf.ByteString
          getNameBytes() {
        java.lang.Object ref = name_;
        if (ref instanceof String) {
          com.google.protobuf.ByteString b = 
              com.google.protobuf.ByteString.copyFromUtf8(
                  (java.lang.String) ref);
          name_ = b;
          return b;
        } else {
          return (com.google.protobuf.ByteString) ref;
        }
      }
      /**
       * <code>string name = 3;</code>
       * @param value The name to set.
       * @return This builder for chaining.
       */
      public Builder setName(
          java.lang.String value) {
        if (value == null) { throw new NullPointerException(); }
        name_ = value;
        bitField0_ |= 0x00000004;
        onChanged();
        return this;
      }
      /**
       * <code>string name = 3;</code>
       * @return This builder for chaining.
       */
      public Builder clearName() {
        name_ = getDefaultInstance().getName();
        bitField0_ = (bitField0_ & ~0x00000004);
        onChanged();
        return this;
      }
      /**
       * <code>string name = 3;</code>
       * @param value The bytes for name to set.
       * @return This builder for chaining.
       */
      public Builder setNameBytes(
          com.google.protobuf.ByteString value) {
        if (value == null) { throw new NullPointerException(); }
        checkByteStringIsUtf8(value);
        name_ = value;
        bitField0_ |= 0x00000004;
        onChanged();
        return this;
      }

      private int fragment_ ;
      /**
       * <code>int32 fragment = 4;</code>
       * @return The fragment.
       */
      @java.lang.Override
      public int getFragment() {
        return fragment_;
      }
      /**
       * <code>int32 fragment = 4;</code>
       * @param value The fragment to set.
       * @return This builder for chaining.
       */
      public Builder setFragment(int value) {
        
        fragment_ = value;
        bitField0_ |= 0x00000008;
        onChanged();
        return this;
      }
      /**
       * <code>int32 fragment = 4;</code>
       * @return This builder for chaining.
       */
      public Builder clearFragment() {
        bitField0_ = (bitField0_ & ~0x00000008);
        fragment_ = 0;
        onChanged();
        return this;
      }
      @java.lang.Override
      public final Builder setUnknownFields(
          final com.google.protobuf.UnknownFieldSet unknownFields) {
        return super.setUnknownFields(unknownFields);
      }

      @java.lang.Override
      public final Builder mergeUnknownFields(
          final com.google.protobuf.UnknownFieldSet unknownFields) {
        return super.mergeUnknownFields(unknownFields);
      }


      // @@protoc_insertion_point(builder_scope:com.webank.ai.eggroll.api.storage.StorageLocator)
    }

    // @@protoc_insertion_point(class_scope:com.webank.ai.eggroll.api.storage.StorageLocator)
    private static final com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator DEFAULT_INSTANCE;
    static {
      DEFAULT_INSTANCE = new com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator();
    }

    public static com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator getDefaultInstance() {
      return DEFAULT_INSTANCE;
    }

    private static final com.google.protobuf.Parser<StorageLocator>
        PARSER = new com.google.protobuf.AbstractParser<StorageLocator>() {
      @java.lang.Override
      public StorageLocator parsePartialFrom(
          com.google.protobuf.CodedInputStream input,
          com.google.protobuf.ExtensionRegistryLite extensionRegistry)
          throws com.google.protobuf.InvalidProtocolBufferException {
        Builder builder = newBuilder();
        try {
          builder.mergeFrom(input, extensionRegistry);
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
          throw e.setUnfinishedMessage(builder.buildPartial());
        } catch (com.google.protobuf.UninitializedMessageException e) {
          throw e.asInvalidProtocolBufferException().setUnfinishedMessage(builder.buildPartial());
        } catch (java.io.IOException e) {
          throw new com.google.protobuf.InvalidProtocolBufferException(e)
              .setUnfinishedMessage(builder.buildPartial());
        }
        return builder.buildPartial();
      }
    };

    public static com.google.protobuf.Parser<StorageLocator> parser() {
      return PARSER;
    }

    @java.lang.Override
    public com.google.protobuf.Parser<StorageLocator> getParserForType() {
      return PARSER;
    }

    @java.lang.Override
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator getDefaultInstanceForType() {
      return DEFAULT_INSTANCE;
    }

  }

  private static final com.google.protobuf.Descriptors.Descriptor
    internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_descriptor;
  private static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\033eggroll/storage-basic.proto\022!com.weban" +
      "k.ai.eggroll.api.storage\"\201\001\n\016StorageLoca" +
      "tor\022<\n\004type\030\001 \001(\0162..com.webank.ai.eggrol" +
      "l.api.storage.StorageType\022\021\n\tnamespace\030\002" +
      " \001(\t\022\014\n\004name\030\003 \001(\t\022\020\n\010fragment\030\004 \001(\005*?\n\013" +
      "StorageType\022\014\n\010LEVEL_DB\020\000\022\r\n\tIN_MEMORY\020\001" +
      "\022\010\n\004LMDB\020\002\022\t\n\005REDIS\020\003b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
        });
    internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_com_webank_ai_eggroll_api_storage_StorageLocator_descriptor,
        new java.lang.String[] { "Type", "Namespace", "Name", "Fragment", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}