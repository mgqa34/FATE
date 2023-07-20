/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.osx.core.flow;

import java.util.List;

public class NamespaceFlowProperty<T> {

    private final String namespace;
    private final Property<List<T>> property;
    private final PropertyListener<List<T>> listener;

    public NamespaceFlowProperty(String namespace,
                                 Property<List<T>> property,
                                 PropertyListener<List<T>> listener) {
        this.namespace = namespace;
        this.property = property;
        this.listener = listener;
    }

    public Property<List<T>> getProperty() {
        return property;
    }

    public String getNamespace() {
        return namespace;
    }

    public PropertyListener<List<T>> getListener() {
        return listener;
    }
}